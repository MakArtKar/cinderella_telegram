Репозиторий построен из темплейта [отсюда](https://github.com/ashleve/lightning-hydra-template/).

* `src/models/components/text_linear_model.py` - простая линейная модель с несколькими слоями после усреднения эмбедингов
* `src/models/tg_messages_module.py` - использует переданную модель для обучения ничего интересного :)
* `src/datamodules/components/df_dataset.py` - датасет, возвращающий сэмплы и принимающий датафрейм
* `src/datamodules/tg_messages_datamodule.py` - скачивает, обрабатывает и инициализирует датасет
* `src/utils/collators.py` - коллатор для датасета

[Отчет](https://wandb.ai/makartkar/nlp-hw1/reports/NLP-HW-1--VmlldzoyODMxODIx) по обучению из WandB (train разбил 9 к 1 на train и val)
