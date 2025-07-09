# 🔄 IDE/Editor Cache Solutions

## For VS Code / Cursor:
1. **Reload Window**: `Ctrl+Shift+P` → "Developer: Reload Window"
2. **Clear Workspace Cache**: Close and reopen the workspace
3. **Force Refresh**: `Ctrl+Shift+P` → "Developer: Refresh"
4. **Restart Extension Host**: `Ctrl+Shift+P` → "Developer: Restart Extension Host"

## For PyCharm:
1. **Invalidate Caches**: `File` → `Invalidate Caches and Restart`
2. **Clear System Cache**: `File` → `Invalidate Caches` → Select all → `Invalidate and Restart`
3. **Refresh Project**: `File` → `Synchronize` or `Ctrl+Alt+Y`

## For Sublime Text:
1. **Reload File**: `File` → `Revert File`
2. **Clear Cache**: Close file, delete `.sublime-workspace`, reopen
3. **Force Refresh**: `View` → `Toggle Setting` → `reload_file_on_change`

## For Vim/Neovim:
1. **Reload File**: `:e!` (force reload)
2. **Clear Buffer**: `:bd` then reopen file
3. **Refresh All**: `:bufdo e!`

## For Atom:
1. **Reload Window**: `Ctrl+Shift+F5`
2. **Clear Cache**: `File` → `Settings` → `Packages` → Disable/Enable relevant packages
3. **Refresh Tree**: `Ctrl+Shift+P` → "Tree View: Refresh"

## Generic Solutions:
1. **Close and reopen the file**
2. **Restart your IDE/Editor completely**
3. **Clear browser cache if using web-based editor**
4. **Check if file is read-only or locked**

## Command Line Verification:
```bash
# Always verify the actual file contents
cat .env | grep EXCEL_FILE_PATH

# Force refresh with Python
python -c "from dotenv import load_dotenv; load_dotenv(override=True); import os; print(os.getenv('EXCEL_FILE_PATH'))"
```