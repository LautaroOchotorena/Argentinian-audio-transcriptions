import pandas as pd

female_df = pd.read_csv("./data/line_index_female.tsv", sep='\t',
                        names=['audio_path', 'transcription'])
male_df = pd.read_csv("./data/line_index_male.tsv", sep='\t',
                      names=['audio_path', 'transcription'])

if __name__ == '__main__':
    import librosa
    import os

    print('Female df:')
    print(female_df.head(), '\n')

    print('Male df:')
    print(male_df.head(), '\n')

    if os.path.exists('./data/female_df.csv'):
        female_df = pd.read_csv('./data/female_df.csv')

    else:
        folder_path_female = r'./data/female_audio'

        for index, filename in enumerate(female_df['audio_path']):
            file_path = os.path.join(folder_path_female, filename) + '.wav'
            y, sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            female_df.loc[index, 'first_path'] = 'female_audio'
            female_df.loc[index, 'sr'] = sr
            female_df.loc[index, 'duration'] = duration
        female_df['transcription_len'] = female_df['transcription'].apply(len)
        
        female_df.to_csv('./data/female_df.csv', index=False)
    
    if os.path.exists('./data/male_df.csv'):
        male_df = pd.read_csv('./data/male_df.csv')
    
    else:
        folder_path_male = r'./data/male_audio'

        for index, filename in enumerate(male_df['audio_path']):
            file_path = os.path.join(folder_path_male, filename) + '.wav'
            y, sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            male_df.loc[index, 'first_path'] = 'male_audio'
            male_df.loc[index, 'sr'] = sr

            male_df.loc[index, 'duration'] = duration

        male_df['transcription_len'] = male_df['transcription'].apply(len)
        male_df.to_csv('./data/male_df.csv', index=False)
    
    print('Female audio descriptive statistics:')
    print(female_df.describe([0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.98]), '\n')

    print('Male audio descriptive statistics:')
    print(male_df.describe([0.25, 0.5, 0.75, 0.85, 0.9, 0.95, 0.98]), '\n')

    print('Combined dataframe saved')
    df = pd.concat([female_df, male_df], ignore_index=True)
    df.to_csv('./data/df.csv', index=False)