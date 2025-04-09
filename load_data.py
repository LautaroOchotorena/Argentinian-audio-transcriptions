import pandas as pd

female_df = pd.read_csv("./data/line_index_female.tsv", sep='\t',
                        names=['audio_path', 'transcription'])
male_df = pd.read_csv("./data/line_index_male.tsv", sep='\t',
                      names=['audio_path', 'transcription'])

if __name__ == '__main__':
    print('Female df:')
    print(female_df.head(), '\n\n')

    print('Male df:')
    print(male_df.head())