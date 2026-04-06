# Новый датасет: [название_датасета]

## Структура данных
```
data/
├── [dataset_name]/
│   ├── metadata.csv
│   ├── mix_0000/
│   │   ├── mixture.wav
│   │   └── sources/
│   │       ├── s1.wav
│   │       └── s2.wav
│   └── ...
```

## Конфигурационные файлы

### configs/data/[dataset_name].yaml
```yaml
defaults:
  - feature_schema: [dataset_name].yaml
  - target_schema: [dataset_name].yaml
  - raw_data: [dataset_name].yaml
  - collator: dynamic_collator.yaml
  - _self_

_target_: src.data.datamodule.AudioDataModule

audio_path_key: audio_path
target_sr: 16000

mel_spectrogram_cfg:
  _recursive_: false
  _target_: src.utils.instantiators.skip_insantiation_helper
  obj:
    _target_: src.data.components.preprocessing.audio.MelSpectrogram
    sample_rate: ${data.target_sr}
    n_mels: 128
    n_fft: 1024
    hop_length: 512
    f_min: 0.0
    f_max: null
```

### configs/data/feature_schema/[dataset_name].yaml
```yaml
_target_: src.data.components.containers.FeatureSchema

numerical:
  _target_: src.data.components.containers.NumericalFeatureInfo
  feature_names:
    - mel_spectrogram
  torch_dtype:
    _target_: src.utils.utils.torch_dtype
    dtype: float32
```

### configs/data/target_schema/[dataset_name].yaml
```yaml
_target_: src.data.components.containers.TargetSchema

categorical:
  _target_: src.data.components.containers.CategoricalFeatureInfo
  feature_names:
    - class_id  # или другие таргеты в зависимости от задачи
  torch_dtype:
    _target_: src.utils.utils.torch_dtype
    dtype: long
  vocabularies_size:
    - [число_классов]
  embeddings_dim:
    - [размер_эмбеддингов]
```

### configs/data/raw_data/[dataset_name].yaml
```yaml
_target_: src.data.datamodule.RawData
_recursive_: false

dataset_dir: ${paths.data_dir}/[dataset_name]/

feature_data_cfg:
  _target_: src.data.components.raw_data.PickleReader
  data_source: ${paths.data_dir}/[dataset_name]/features.pkl

target_data_cfg:
  _target_: src.data.components.raw_data.PickleReader
  data_source: ${paths.data_dir}/[dataset_name]/targets.pkl

train_keys_cfg:
  _target_: src.data.components.raw_data.PickleReader
  data_source: ${paths.data_dir}/[dataset_name]/train_keys.pkl

val_keys_cfg:
  _target_: src.data.components.raw_data.PickleReader
  data_source: ${paths.data_dir}/[dataset_name]/val_keys.pkl

test_keys_cfg:
  _target_: src.data.components.raw_data.PickleReader
  data_source: ${paths.data_dir}/[dataset_name]/test_keys.pkl
```

## Скрипт обработки данных

Создайте `notebooks/[dataset_name].ipynb` или `src/data/process_[dataset_name].py` для:

1. Загрузки/подготовки сырых данных
2. Создания features.pkl, targets.pkl, train/val/test_keys.pkl
3. Форматирования в структуру source_sep_formatted

## DVC интеграция

1. Добавьте новый датасет в DVC:
```bash
dvc add data/[dataset_name]
```

2. Обновите .dvc файлы в git

## Запуск

```bash
# Обучение на новом датасете
python src/train.py data=[dataset_name]

# Оценка
python src/eval.py data=[dataset_name]
```