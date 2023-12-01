#!/bin/bash


cd data_prepared

# delte the folders 5_rs and all_rs
# -type d:                               select only directories
# "*rs*"                                 Directories to delete
# -exec rm -r {} \:                   delete directory if found
#find . -type d -name "*rs*" -exec rm -r {} \;
#echo "directories *rs* removed"

## PLUS dataset should only consider 003 and 004 synthetic subdirectories
## min depth 2 to not delete directories in the current directory

datasets=("PLUS" "IDIAP" "PROTECT" "SCUT")
datasets=("PROTECT")
# Loop over the array
for db in "${datasets[@]}"; do
  echo $db " dataset:"
  cd $db
  if [ "$db" == "PLUS" ]; then
      find . -mindepth 2 -maxdepth 2 -type d ! \( -name "003" -o -name "004" \) -exec rm -r -- {}/ \;
      echo "All variants except 003 and 004 deleted"
  fi
  if [ "$db" == "SCUT" ]; then
      find . -mindepth 2 -maxdepth 2 -type d ! \( -name "007" -o -name "008" \) -exec rm -r -- {}/ \;
      echo "All variants except 007 and 008 deleted"
  fi
  if [ "$db" == "IDIAP" ]; then
      find . -mindepth 2 -maxdepth 2 -type d ! \( -name "009" \) -exec rm -r -- {}/ \;
      echo "All variants except 009 deleted"
  fi

  ## move images to variant folder
  find . -maxdepth 1 -mindepth 1 -type d | while read dir; do
    # dir = typ (genuine, spoofed, ...)
    find $dir -maxdepth 1 -mindepth 1 -type d | while read subdir; do
      # subdir = 003 and 004
      cd $subdir
      find  . -maxdepth 1 -mindepth 1 -type d | while read subsubdir; do
        find . -mindepth 2 -type f -exec mv --backup=numbered -t . {} +
      done
      cd ../..
    done
  done

  #remove empty folders
  find . -mindepth 1 -type d -name "reference" | while read dir; do
    if [ -z "$(ls -A "$dir")" ]; then
      parentFolder="${dir%/*}"
      rm -r $parentFolder
    fi
  done

  # rename .png~1~ files to _copy1.png
  find . -type f -name '*~' -print0 | while IFS= read -r -d '' file; do
    # Extract the directory path, filename, and extension
    filename="${file%*/}"

    # Extract the numeric value after tilde
    numeric_suffix="${file#*~}"
    numeric_suffix="${numeric_suffix%%~*}"

    # Remove the numeric suffix and the tilde
    filename="${file%.png*}"

    # Append "_copy" and the numeric suffix before the extension
    new_filename="${filename}_copy${numeric_suffix}.png"

    # Rename the file
    mv "$file" "$new_filename"
  done
  echo "All images in subfolders moved to variant folders "

  # copy each variant of the spoofed synthetic to a own folder
  # spoofed_synthethic_drit with the two subfolders 003 and 004 into spoofed_synthethic_drit_003 and spoofed_synthetic_drit_004
  find . -maxdepth 1 -mindepth 1 -type d | while read dir; do
    # dir = typ (genuine, spoofed, ...)

    find $dir -maxdepth 1 -mindepth 1 -type d | while read subdir; do
      # subdir = 003 and 004
      newFolder=$(echo "${subdir#./}" | sed 's/\//_/g')
      variantNumber=${subdir##*/}

      cp -r $dir $newFolder
      cd $newFolder

      #remove all other folders in the created directory
      find . -maxdepth 1 -type d ! -name "$variantNumber" ! -name "." -exec rm -r {} +

      # move images to this directory and remove variant folder
      mv $variantNumber/* ./
      rm -r $variantNumber
      cd ..
    done
  done

  #delte synthethic folders with subfolders in it
  shopt -s nullglob
  for dir in */; do
    # Check if the current directory has subdirectories
    subdirs=("$dir"*/)
    if [ ${#subdirs[@]} -gt 0 ]; then
      rm -r "$dir"
    fi
  done
  echo "Created for every variant own class"
  echo $(pwd)
  cd ..
  echo $(pwd)
done