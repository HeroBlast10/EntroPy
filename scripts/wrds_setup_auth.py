"""One-time WRDS credential setup. Run this in YOUR OWN TERMINAL.

    python scripts/wrds_setup_auth.py

The script asks for your WRDS username and password (typed, not echoed) and
writes them to PostgreSQL's standard credential file. Nothing is printed, echoed
or written to the repository.

Why this exists
---------------
``wrds.Connection`` prompts interactively for a username when it cannot find
one, which deadlocks any unattended pull. libpq itself reads ``pgpass``, so
writing it once means every later command runs without ever handling secrets.

Never commit credentials. This repository is public; ``.gitignore`` excludes
both the pgpass location and ``config/wrds_credentials.json``. If you prefer
environment variables instead::

    setx WRDS_USERNAME your_username
    setx WRDS_PASSWORD your_password
"""

from __future__ import annotations

import getpass
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.wrds_ingest.client import pgpass_path


def main() -> int:
    print("WRDS credential setup for EntroPy")
    print("-" * 60)
    print("Your password is not echoed, printed, or written to the repository.")

    username = (getpass.getpass("WRDS username: ").strip()) or os.environ.get("WRDS_USERNAME", "")
    if not username:
        print("error: no username provided", file=sys.stderr)
        return 2
    password = getpass.getpass("WRDS password: ")
    if not password:
        print("error: no password provided", file=sys.stderr)
        return 2

    target = pgpass_path()
    if target is None:
        print(
            "error: cannot locate pgpass; set PGPASSFILE or APPDATA/HOME",
            file=sys.stderr,
        )
        return 2

    target.parent.mkdir(parents=True, exist_ok=True)
    line = (
        f"wrds-pgdata.wharton.upenn.edu:9737:wrds:{username}:{password}\n"
    )
    target.write_text(line, encoding="utf-8")

    # libpq refuses to honour a pgpass readable by group or other.
    try:
        os.chmod(target, 0o600)
    except OSError:
        print(
            f"  note: could not restrict permissions on {target} (unusual on Windows); "
            "libpq may still read it."
        )
    print()
    print(f"  Wrote {target}")
    print()
    print("Verify with:")
    print("    python scripts/wrds_pull.py probe")
    return 0


if __name__ == "__main__":
    sys.exit(main())
