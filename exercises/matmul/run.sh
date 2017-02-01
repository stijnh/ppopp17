make all && prun -np 1 -v -native '-C TitanX --gres=gpu:1' "$@"
