files=$(find .. -name "*.py")
for f in $files ; do echo $f ; 2to3 -w $f; done
echo "Done!"
