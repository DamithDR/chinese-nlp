import pandas as pd

df1 = pd.read_excel('data/data_file.xlsx', sheet_name='Com_species_corpus_500')
df2 = pd.read_excel('data/data_file.xlsx', sheet_name='Sci_species_corpus_1500')

df = pd.concat([df1, df2])

df = df.drop('sentence', axis=1)

df.to_csv('data/processed/annotation.tsv', sep='\t', index=False)
