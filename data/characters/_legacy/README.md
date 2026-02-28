# _legacy

Zdes lezhit sovmestimyy sloy staryh personality-profiley.

- `personality_profiles/*.json` - legacy-profiles dlya `core/personality_engine.py`.
- `default/` - polnotsennyy legacy-character po strukture `data/characters/asya/`.
- prompty dlya etih profiley teper v `prompts/legacy/system/*.txt`.

Naznachenie:
1) ne lomat starye komandy `/persona ...`;
2) imet migratsionnyy etalon `default` v novoy character-strukture.

Plan migratsii:
- profile-driven logiku perenosit v `data/characters/<id>/...`;
- posle migratsii udalyat `_legacy/personality_profiles`.
