#!/bin/bash

main() {
    if [[ ${#@} -eq 0 ]]
    then
        to_archive=(figures
                    plate_model
                    prepared_data
                    source_data
                    supplementary_figures)
    else
        to_archive=("$@")
    fi

    for directory in "${to_archive[@]}"
    do
        zip \
            -r \
            "${directory}.zip" \
            "${directory}" \
            -x '*/.*' \
            || break
    done
    return 0
}

main "$@"
