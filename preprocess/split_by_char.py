import pyperclip

with open('split.txt', 'r', encoding='utf-8') as f:
    txt = f.read()

txt = txt.replace('\n', ' ')
txt = txt.replace(';', '\n')

# Colocar txt na area de transferência
pyperclip.copy(txt)
print('ok')
