for filename in *;
    do
        sed -i 's/main/main/g' "$filename";
    done
