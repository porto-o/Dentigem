from pathlib import Path
import re

def _read_patient_data(filepath: str):
    try:
        path = Path(filepath)
        if not path.exists() or not path.is_file():
            return None

        def clean(text):
            return re.sub(r"[#*`\-_\[\]\(\)]", "", text).strip()

        content = f"=== PATIENT RECORD ===\n{clean(path.read_text(encoding='utf-8'))}"

        patient_name = path.stem
        sessions_dir = path.parent.parent / "Sessions"

        if sessions_dir.exists():
            session_files = list(sessions_dir.glob(f"{patient_name}*.md"))
            for session_file in sorted(session_files):
                session_text = clean(session_file.read_text(encoding="utf-8"))
                content += f"\n\n=== SESSION: {session_file.name} ===\n{session_text}"

        return content

    except Exception as e:
        print(f"Error reading file {e}")
        return None