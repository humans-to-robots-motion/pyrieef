if [ -z "$1" ]
  then
    echo "No argument supplied"
    exit 1
fi
files=$(find .. -name "*.${1}")
for f in $files ; do echo $f ; rm $f; done
echo "Done."
