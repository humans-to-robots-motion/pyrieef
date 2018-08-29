while IFS= read -r file ; do rm -- "$file" ; done < pyc_files.txt
