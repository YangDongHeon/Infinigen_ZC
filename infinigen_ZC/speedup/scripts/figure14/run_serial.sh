FLEXGEN_PATH=/home/ipa/dongheon/infinigen/speedup/flexgen
for SCHEME in "infinigen_ZC"
do
  echo "Processing scheme: $SCHEME serial"
  INFINIGEN_PATH=/home/ipa/dongheon/$SCHEME/speedup/flexgen/infinigen
  rm $FLEXGEN_PATH/flexgen/flex_opt.py
  rm $FLEXGEN_PATH/flexgen/pytorch_backend.py
  cp $INFINIGEN_PATH/flex_opt_serial.py $FLEXGEN_PATH/flexgen/flex_opt.py
  cp $INFINIGEN_PATH/pytorch_backend.py $FLEXGEN_PATH/flexgen/pytorch_backend.py

  for BSZ in 4
  do
    CMD="--model huggingface/opt-6.7b --percent 100 0 0 100 100 0 --overlap false --gpu-batch-size $BSZ --num-gpu-batches 1 --prompt-len 1920 --gen-len 128 --warmup-input-path /home/ipa/dongheon/infinigen/speedup/scripts/figure14/pg19_firstbook.txt --test-input-path /home/ipa/dongheon/infinigen/speedup/scripts/figure14/pg19_firstbook.txt"
    python -m flexgen.flex_opt $CMD
    #nsys profile -o infinigen_ZC_serial -t cuda,nvtx --force-overwrite true python -m flexgen.flex_opt $CMD
  done

  #for PROMPT_LEN in 384 896 1408 1920
  #do
  #  CMD="--model huggingface/opt-6.7b --percent 100 0 0 100 100 0 --overlap false --gpu-batch-size 8 --num-gpu-batches 1 --prompt-len $PROMPT_LEN --gen-len 128 --warmup-input-path /home/ipa/dongheon/infinigen/speedup/scripts/figure14/pg19_firstbook.txt --test-input-path /home/ipa/dongheon/infinigen/speedup/scripts/figure14/pg19_firstbook.txt"
  #  python -m flexgen.flex_opt $CMD
  #done
done