`python3 -c "import readme2tex"` || sudo -H pip3 install --upgrade readme2tex

for md in ./*.md
do
  echo "$md -> $md"
  python3 -m readme2tex --output $md $md
done
