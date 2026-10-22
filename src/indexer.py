from pathlib import Path

from src.chunking.base import BaseChunker, Chunk
from src.chunking.code_chunker import CodeChunker
from src.chunking.text_chunker import TextChunker
from dataclasses import dataclass
import bm25s
import pickle
from tqdm import tqdm

import hashlib
import json


@dataclass
class BM25Index:
    """
    A BM25 retriever bundled with the chunks it was built from.
    """
    retriever: bm25s.BM25
    chunks: list[Chunk]


class Indexer:
    """
    Run the indexing pipeline from raw files to a persisted index.
    """
    def __init__(
        self,
        raw_dir: str,
        processed_dir: str,
        max_chunk_size: int = 2000,
    ) -> None:
        """Configure paths, chunk size and the extension-to-chunker mapping.

        Args:
            raw_dir: Directory containing the corpus to index.
            processed_dir: Directory where the index is saved.
            max_chunk_size: Maximum characters per chunk.
        """

        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.max_chunk_size = max_chunk_size

        self.code_chunker: BaseChunker = CodeChunker()
        self.text_chunker: BaseChunker = TextChunker()

        self.extension_map: dict[str, BaseChunker] = {
            ".py": self.code_chunker,
            ".md": self.text_chunker,
            ".txt": self.text_chunker,
        }  # add ".txt"

    def discover_files(self) -> list[Path]:
        """List the .py and .md files under raw_dir, sorted.

        Returns:
            Sorted file paths, ignoring __pycache__, .git and node_modules.

        Raises:
            FileNotFoundError: If raw_dir does not exist.
        """
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw directory not found:{self.raw_dir}")

        excluded_dirs = {"__pycache__", ".git", "node_modules"}

        files = [
            path
            for path in self.raw_dir.rglob("*")
            if path.is_file()
            and path.suffix in self.extension_map
            and not excluded_dirs.intersection(path.parts)
        ]

        return sorted(files)

    def read_file(self, path: Path) -> str | None:
        """Read a file as UTF-8 text.

        Args:
            path: File to read.

        Returns:
            The file content.

        Raises:
            UnicodeDecodeError: If the file is not valid UTF-8.
            OSError: If the file cannot be read.
        """
        return path.read_text(encoding="utf-8")

    def route_file(self, path: Path) -> BaseChunker | None:
        """Pick the chunker matching a file extension.

        Args:
            path: File to route.

        Returns:
            The matching chunker, or None if the extension is unsupported.
        """
        return self.extension_map.get(path.suffix)

    def chunk_file(
        self, path: Path, content: str, chunker: BaseChunker
    ) -> list[Chunk]:
        """Split a file's content into chunks with the given chunker.

        Args:
            path: File the content comes from.
            content: Raw file content.
            chunker: Chunker used to split it.

        Returns:
            Chunks carrying their file path and character range.
        """

        return chunker.chunk(
            content=content,
            file_path=str(path),
            max_chunk_size=self.max_chunk_size,
        )

    def build_index(self, chunks: list[Chunk]) -> "BM25Index":
        """Build a BM25 index over the text of all chunks.

        Args:
            chunks: Chunks to index.

        Returns:
            The BM25 retriever together with its chunks.
        """
        corpus_texts = [chunk.text for chunk in chunks]
        corpus_tokens = bm25s.tokenize(
            corpus_texts, stopwords="en"
        )  # the is a an and of to

        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        return BM25Index(retriever=retriever, chunks=chunks)

    def save_index(self, index: BM25Index) -> None:
        """Persist the BM25 index and its chunks under processed_dir.

        Args:
            index: Index to save.
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        index.retriever.save(str(self.processed_dir / "bm25_index"))

        with open(self.processed_dir / "chunks.pkl", "wb") as file:
            pickle.dump(index.chunks, file)

    def run(self) -> list[Chunk]:
        """Run the full indexing pipeline.

        Unreadable files are skipped with a warning.

        Returns:
            All chunks that were indexed.

        Raises:
            OSError: If the index cannot be saved.
        """

        files = self.discover_files()

        all_chunks: list[Chunk] = []

        for path in tqdm(files, desc="Chunking"):
            try:
                content = self.read_file(path)
                if content is None:
                    continue
            except (UnicodeDecodeError, OSError) as error:
                print(f"Warning: could not read {path}: {error}")
                continue

            chunker = self.route_file(path)
            if chunker is None:
                continue

            chunks = self.chunk_file(path, content, chunker)
            all_chunks.extend(chunks)

        index = self.build_index(all_chunks)

        try:
            self.save_index(index)
        except OSError as error:
            print(f"Error: could not save index: {error}")
            raise

        print(
            f"Ingestion complete! Indexed {len(all_chunks)} chunks "
            f"under {self.processed_dir}/"
        )

        return all_chunks

    # part dail Incremental Indexing

    def _compute_file_hash(self, path: Path, content: str) -> str:
        """Return the MD5 hash of a file's content, used to detect changes.

        Args:
            path: File the content comes from.
            content: File content to hash.

        Returns:
            Hexadecimal digest.
        """
        return hashlib.md5(content.encode("utf-8")).hexdigest()

    def _load_state(self) -> dict[str, str]:
        """Load the saved mapping of file path to content hash.

        Returns:
            The state, or an empty dict if none was saved yet.
        """
        state_path = self.processed_dir / "state.json"
        if not state_path.exists():
            return {}

        with state_path.open("r", encoding="utf-8") as file:
            from typing import cast
            return cast(dict[str, str], json.load(file))

    def _save_state(self, state: dict[str, str]) -> None:
        """Save the mapping of file path to content hash.

        Args:
            manifest: Hashes of the files seen in the current run.
        """
        state_path = self.processed_dir / "state.json"
        with state_path.open("w", encoding="utf-8") as file:
            json.dump(state, file, indent=2)

    def run_incremental(self) -> list[Chunk]:
        """Re-chunk only new or changed files and rebuild the index (bonus).

        Chunks of unchanged files are reused from the previous run.

        Returns:
            All chunks of the updated index.
        """
        old_state = self._load_state()
        new_state: dict[str, str] = {}

        files = self.discover_files()
        chunks_loaded = False
        try:
            with (self.processed_dir / "chunks.pkl").open("rb") as file:
                import pickle

                existing_chunks: list[Chunk] = pickle.load(file)
                chunks_loaded = True
        except (FileNotFoundError, OSError):
            existing_chunks = []

        unchanged_chunks = []
        # path : list[chunks]
        chunks_by_file: dict[str, list[Chunk]] = {}

        # hna ghadi najme3 ga3 chunks ou ndirhom f dict
        # ghadi ywali key hio pathfile ou value hio chunks dialo li mno.
        for chunk in existing_chunks:
            chunks_by_file.setdefault(chunk.file_path, []).append(chunk)

        all_chunks: list[Chunk] = []  # unchanged chunks + new chunks
        files_to_process = []  # changed files and new files

        for path in files:
            try:
                content = self.read_file(path)
                if content is None:
                    continue
            except (UnicodeDecodeError, OSError) as error:
                print(f"Warning: could not read {path}: {error}")
                continue

            file_hash = self._compute_file_hash(path, content)
            new_state[str(path)] = file_hash

            if old_state.get(str(path)) == file_hash and chunks_loaded:
                unchanged_chunks.extend(
                    chunks_by_file.get(str(path), [])
                )  # we stor here chunks it is not changes
            else:
                files_to_process.append((path, content))

        print(
            f"Incremental: {len(files_to_process)} changed/new files, "
            f"{len(files) - len(files_to_process)} unchanged."
        )

        all_chunks.extend(unchanged_chunks)

        # we chunks only new/changed file here.
        for path, content in tqdm(
            files_to_process, desc="Chunking (incremental)"
        ):
            chunker = self.route_file(path)
            if chunker is None:
                continue
            chunks = self.chunk_file(path, content, chunker)
            all_chunks.extend(chunks)

        index = self.build_index(
            all_chunks
        )  # crache here if file (chunks.pkl not found)
        self.save_index(index)
        self._save_state(new_state)

        print(
            f"Incremental indexing complete! Indexed {len(all_chunks)} chunks."
        )

        return all_chunks


# bhal haka katkhdem rglob.

# raw/
# ├── a.py
# ├── b.md
# └── folder1/
#     ├── c.py
#     └── folder2/
#         ├── d.md
#         └── e.txt

# ghado djib ->

# a.py
# b.md
# folder1
# folder1/c.py
# folder1/folder2
# folder1/folder2/d.md
# folder1/folder2/e.txt
