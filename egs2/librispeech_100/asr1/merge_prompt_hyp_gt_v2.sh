# Read path of experiment folder. Assert that $1 is dir.
if [[ -d "$1" && $2 == ?(-)+([0-9]) && -r "$3" && $4 == ?(-)+([0-9]) ]]
then
  echo "Reading folder: $1"
  echo "Number of first lines: $2"
  echo "GT path is : $3"
  echo "Number of subwords unit that are cut off for prompt: $4"

  dirp=$1
  # File 1: 
  hyp="${dirp}/text"

  # Source gt text:
  gt="$3"

  # Create prompt file
  SCRIPT_DIR=$(dirname "$0")
  echo "Script dir is: $SCRIPT_DIR"
  prompt_file="${dirp}/prompt_head_$2.txt"
  head -n $2 ${gt} | spm_encode --model="${SCRIPT_DIR}/data/en_token_list/bpe_unigram5000/bpe.model" | sed -r -e "s/(\s[A-Za-z0-9▁]+){$4}$//" | spm_decode --model="${SCRIPT_DIR}/data/en_token_list/bpe_unigram5000/bpe.model" > $prompt_file

  exec 6<"$gt"
  exec 7<"$hyp"
  while read -r prompt_line
  do
    read -r gt_line <&6
    read -r hyp_line <&7
    echo "Prompt: ${prompt_line}"
    echo "GT: ${gt_line}"
    echo "Hyp: ${hyp_line}"
    echo
  done <"${prompt_file}"
  exec 6<&-
  exec 7<&-
else
  echo -e "USAGE: arg1: dir of exp folder\narg2: head -n number\narg3: path of gt text\narg4: cutoff"
fi
