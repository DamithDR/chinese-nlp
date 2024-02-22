delimitter = "。"

file1 = 'data/extended_data/Com_species_corpus.txt'
file2 = 'data/extended_data/sci_species_corpus.txt'

lines = []

with open(file1, 'r') as f1:
    lines.extend(f1.readlines())

with open(file2, 'r') as f2:
    lines.extend(f2.readlines())

splitted_lines = []

print(len(lines))

for line in lines:
    split = str(line).split(delimitter)

    for s in split:
        s = s.strip()
        if s != '':
            splitted_lines.append(s)

splitted_lines = [f'{line}{delimitter}' for line in splitted_lines]
splitted_lines = [line.replace('\n', '') for line in splitted_lines]

print(len(splitted_lines))

with open('data/extended_data/combined.txt', 'w') as combined:
    combined.writelines("\n".join(splitted_lines))
