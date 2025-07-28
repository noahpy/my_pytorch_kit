
python -m build
if [ $? -ne 0 ]; then
    exit 1
fi
twine check dist/*
if [ $? -ne 0 ]; then
    exit 1
fi
twine upload dist/*
