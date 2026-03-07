import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ParsedSession:
    meta: Dict[str, str]
    scalars: Dict[str, float]
    arrays: Dict[str, List[float]]
    filename: str
    raw_block: str

class MedPCParser:
    def __init__(self):
        self.skipped_sessions: List[Tuple[str, str, str]] = []  # (filename, reason, snippet)

    def parse_file(self, content: str, filename: str) -> List[ParsedSession]:
        """
        Main entry point: split file into session blocks and parse each one.
        """
        session_blocks = self._extract_session_blocks(content)
        parsed = []

        for block in session_blocks:
            try:
                session = self._parse_single_session(block.strip(), filename)
                if session:
                    parsed.append(session)
            except Exception as e:
                short = (block[:180] + "...") if len(block) > 180 else block
                self.skipped_sessions.append((filename, str(e), short.replace("\n", " ")))

        return parsed

    def _extract_session_blocks(self, content: str) -> List[str]:
        """
        Split raw MedPC export text into individual session blocks.
        Looks for common delimiters: \\filename, Start Date:, ====== , blank lines + new MSN, etc.
        """
        blocks = []
        current_lines = []
        lines = content.splitlines()

        for line in lines:
            stripped = line.strip()

            # New session indicators
            if (
                stripped.startswith("\\") and "MPC" in stripped.upper()  # \\PROGRAMNAME.MPC
                or stripped.startswith("Start Date:")
                or re.match(r"={5,}", stripped)  # ===== separator
                or (stripped == "" and current_lines and "Start Date:" in "".join(current_lines))
            ):
                if current_lines:
                    block = "\n".join(current_lines).strip()
                    if len(block) > 150 and "Start Date:" in block:
                        blocks.append(block)
                current_lines = [line]
            else:
                current_lines.append(line)

        # Don't forget the last block
        if current_lines:
            block = "\n".join(current_lines).strip()
            if len(block) > 150 and "Start Date:" in block:
                blocks.append(block)

        return blocks

    def _parse_single_session(self, block: str, filename: str) -> Optional[ParsedSession]:
        lines = block.splitlines()
        meta = {}
        scalars = {}
        arrays = {}
        i = 0

        # 1. Metadata / Header (Subject, MSN, Start Date, Box, Room, etc.)
        while i < len(lines) and not re.match(r"^[A-Z]:\s", lines[i].strip()):
            line = lines[i].strip()
            if ":" in line and not line.startswith("\\"):
                key_part, val_part = line.split(":", 1)
                key = key_part.strip()
                val = val_part.strip()

                # Standard MedPC keys + extras we care about
                known_keys = [
                    "Subject", "MSN", "Start Date", "End Date", "Box", "Room",
                    "Experiment", "Group", "Protocol", "File", "Comment"
                ]
                if key in known_keys or key.lower() in {"box", "room", "cage", "experiment", "group"}:
                    meta[key] = val
            i += 1

        # 2. Scalars (A: 123.45    B: 0    etc.)
        while i < len(lines):
            line = lines[i].strip()
            if re.match(r"^[A-Z]:\s*-?\d+(\.\d+)?", line):
                var, val_str = [part.strip() for part in line.split(":", 1)]
                try:
                    scalars[var] = float(val_str)
                except ValueError:
                    pass
            elif line.startswith("\\") or line.startswith("=" * 5) or not line:
                break  # end of scalars, beginning of arrays or next block
            i += 1

        # 3. Arrays (I: 12.3 45.6 ...    or multi-line)
        current_var = None
        current_data = []

        while i < len(lines):
            line = lines[i].strip()

            # New array header
            m = re.match(r"^([A-Z]):", line)
            if m:
                # Save previous array if any
                if current_var and current_data:
                    arrays[current_var] = current_data

                current_var = m.group(1)
                current_data = []
                # Values may follow on same line
                rest = line.split(":", 1)[1].strip()
                if rest:
                    try:
                        current_data.extend(float(x) for x in rest.split() if x)
                    except ValueError:
                        pass
            # Continuation line (numbers only or with spaces)
            elif current_var and re.match(r"^[-\d\s\.eE]+$", line.replace(" ", "")):
                try:
                    vals = [float(x) for x in line.split() if x.strip("-").replace(".", "").isdigit()]
                    current_data.extend(vals)
                except ValueError:
                    pass

            i += 1

        # Save last array
        if current_var and current_data:
            arrays[current_var] = current_data

        # Require minimal useful data
        if not meta.get("Start Date") or not meta.get("Subject"):
            raise ValueError("Missing required metadata (Start Date or Subject)")

        return ParsedSession(
            meta=meta,
            scalars=scalars,
            arrays=arrays,
            filename=filename,
            raw_block=block[:600] + "..." if len(block) > 600 else block
        )

    def get_skipped_report(self) -> List[Dict]:
        """For logging / display in app."""
        return [
            {"File": f, "Reason": r, "Snippet": s}
            for f, r, s in self.skipped_sessions
        ]