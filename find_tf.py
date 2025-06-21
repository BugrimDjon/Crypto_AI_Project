import os
import sys

# Корневая папка — текущий каталог и site-packages
roots = [os.getcwd()]
try:
    import site
    roots += site.getsitepackages()
except AttributeError:
    pass
roots += sys.path  # добавим пути поиска модулей

seen = set()
print("🔍 Поиск файлов/папок, содержащих 'tensorflow'...\n")
for root in roots:
    if not root or not os.path.exists(root):
        continue
    for dirpath, dirnames, filenames in os.walk(root):
        full = dirpath.lower()
        if "tensorflow" in full:
            if full not in seen:
                print(dirpath)
                seen.add(full)
        for f in filenames:
            if "tensorflow" in f.lower():
                full_path = os.path.join(dirpath, f)
                if full_path not in seen:
                    print(full_path)
                    seen.add(full_path)
