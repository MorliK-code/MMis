import codecs

with codecs.open('tests/test_response_hygiene.py', 'r', 'utf-8-sig') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if 'self.assertEqual(result.text, "Поняла. Перейду сразу к сути.")' in line:
        if 'test_pipeline_replaces_smalltalk_echo' in ''.join(lines[max(0, lines.index(line)-10):lines.index(line)]):
            new_lines.append(line.replace('"Поняла. Перейду сразу к сути."', '"У меня все нормально, спасибо. Как ты?"'))
        elif 'test_pipeline_replaces_generic_short_echo' in ''.join(lines[max(0, lines.index(line)-10):lines.index(line)]):
            new_lines.append(line.replace('"Поняла. Перейду сразу к сути."', '"Поняла. Я на связи и готова помочь. Уточни, что именно нужно."'))
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

with codecs.open('tests/test_response_hygiene.py', 'w', 'utf-8-sig') as f:
    f.writelines(new_lines)
