if [[ -f "$1" && -f "$2" && -f "$3" ]]
then
  echo "All three files: $1, $2 and $3 exist."
  # $1 Prompt
  # $2 Hyp
  # $3 GT

  exec 6<"$2"
  exec 7<"$3"
  while read -r line
  do
    read -r line2 <&6
    read -r line3 <&7
    echo ${line}
    echo ${line3}
    echo ${line2}
    echo
  done <"$1"
exec 6<&-
exec 7<&-
fi
