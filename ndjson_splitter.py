#!/usr/bin/env python3

import json
import os

import pandas as pd


def main():
    print('Processing the combined data file ...')

    dataframe = pd.read_json(
        './data/department_of_justice_press_releases.ndjson',
        lines=True,
        encoding='utf8'
    )

    if not os.path.exists('./data/en/'):
        os.makedirs('./data/en/')

    for index, row in dataframe.iterrows():
        item_id = str(index + 1).zfill(5)

        print(f'Processing file Nr. {str(item_id)} ...')

        item_object = {}
        item_object['id'] = item_id

        for column in row.items():
            cell_heading = column[0]
            cell_value = column[1]

            if cell_heading == 'title':
                item_object['title'] = cell_value

            if cell_heading == 'contents':
                cell_value.replace('\r', '')
                cell_value.replace('\n', ' ')

                item_object['contents'] = ' '.join(cell_value.split())

        item_json_object = json.dumps(
            item_object,
            ensure_ascii=False,
            sort_keys=False,
            indent=4,
            separators=(',', ': ')
        )

        with open('./data/en/' + item_id + '.json', 'w', encoding='utf8') as outfile:
            outfile.write(item_json_object)

    print('')


if __name__ == '__main__':
    main()
