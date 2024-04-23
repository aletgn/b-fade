pip3 uninstall bfade -y
python3 -m build
pip3 install dist/b_fade-0.0.0rc0-py3-none-any.whl
sphinx-apidoc -o ./docs/ ./src/
cd docs/
make clean & make html
cd ..
