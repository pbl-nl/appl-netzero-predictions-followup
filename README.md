## 1) Install Python 3.10 (if not available)

If Python version is 3.14 or higher some modules may not work. For stability 3.10 is recommended.

1. Download Python 3.10 for Windows from the official site:

   - Go to: https://www.python.org/downloads/release/python-310/
   - Download the **Windows installer (64-bit)**.
2. Run the installer and make these choices:

   - ✅ **Install launcher for all users (recommended)** (this installs `py.exe`)
   - ✅ **Add python.exe to PATH** (optional; `py` is the more reliable way)
   - Choose **Customize installation** (optional), but make sure it installs normally.

This will install Python 3.10 *alongside* other versions.

---

## 2) Verify Python 3.10 is available (in VS Code terminal)

Open **VS Code → Terminal → New Terminal**.

### PowerShell

List all installed Python versions known to the launcher:

```powershell
py -0p
```

You should see an entry for `-3.10 ...`.

Confirm 3.10 runs:

```powershell
py -3.10 --version
```

If that prints `Python 3.10.x`, you’re good.

---

## 3) Create a project virtual environment using Python 3.10

In the VS Code terminal, `cd` into your project folder (where you want `.venv`):

```powershell
cd path\to\your\project
```

Create `.venv` with Python 3.10:

```powershell
py -3.10 -m venv .venv
```

---

## 4) Activate the `.venv` in the VS Code terminal

### PowerShell (VS Code default on many systems)

```powershell
.\.venv\Scripts\Activate.ps1
```

If you get an execution policy error, run this once (per terminal session) and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Command Prompt (if your VS Code terminal is cmd.exe)

```bat
.\.venv\Scripts\activate.bat
```

---

## 5) Tell VS Code to use this interpreter (important)

VS Code may not automatically switch to the new `.venv`.

1. Press `Ctrl+Shift+P`
2. Search: **Python: Select Interpreter**
3. Pick the interpreter that looks like:
   - `...\.venv\Scripts\python.exe`

You should see the selected interpreter in the bottom status bar.

---

## 6) Confirm everything is correct

In the activated terminal:

```powershell
python --version
python -c "import sys; print(sys.executable)"
```

You should see Python 3.10.x and a path containing `\.venv\`.

---

## 7) Upgrade pip + install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```


## 8) Deactivate when done

```bash
deactivate
```

---

## Common issues

### `py` is not recognized

- Re-run the Python 3.10 installer and ensure **Python Launcher** is enabled.
- Or install the launcher via the official installer again.

### VS Code keeps using a different Python

- Run **Python: Select Interpreter** and choose the `.venv` one.
- Restart the terminal after selecting it.

### PowerShell activation is blocked

Use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

This is temporary and only affects the current terminal session.
