# КЕЙС 1
# Установим интерактивный режим для отображения графиков
import matplotlib
# Используем интерактивный бэкенд для отображения графиков
# В разных средах могут быть разные бэкенды
# Для Jupyter:
# %matplotlib inline
# Для Colab:
# %matplotlib inline

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# Чтение данных
df = pd.read_csv('1.csv', encoding='utf-8')
print("Структура данных:")
print(df.columns.tolist())
print("\nПервые строки:")
print(df.head())

# 1. Загрузка и предобработка данных
regions_df = df.iloc[3:].reset_index(drop=True)

# 2. Очистка значений
def clean_number(x):
    if isinstance(x, str):
        # Заменяем тире на 0 и убираем пробелы
        cleaned = x.replace('–', '0').replace(' ', '').strip()
        # Если после очистки пустая строка - возвращаем 0
        if cleaned == '' or cleaned == '-':
            return 0.0
        try:
            return float(cleaned)
        except:
            return 0.0
    elif pd.isna(x):
        return 0.0
    return float(x)

# 3. Фильтрация федеральных округов
federal_districts = regions_df[regions_df['Содержание'].str.contains('федеральный округ', na=False)]
district_names = [name.replace(' федеральный округ', '') for name in federal_districts['Содержание']]

# Подготовим данные для графиков
total_population = [clean_number(x) for x in federal_districts.iloc[:, 1]]
male_population = [clean_number(x) for x in federal_districts.iloc[:, 2]]
female_population = [clean_number(x) for x in federal_districts.iloc[:, 3]]

# Данные для России
russia_row = df[df['Содержание'] == 'Российская Федерация']
if not russia_row.empty:
    male_total = clean_number(russia_row.iloc[0, 2])
    female_total = clean_number(russia_row.iloc[0, 3])
else:
    male_total = sum(male_population)
    female_total = sum(female_population)
# График 1: Общая численность населения
plt.figure(figsize=(12, 8))
bars = plt.barh(district_names, total_population, color='skyblue', edgecolor='navy', alpha=0.8)
plt.xlabel('Численность населения', fontsize=12)
plt.title('Общая численность населения по федеральным округам', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)

# Добавляем подписи значений
for bar, value in zip(bars, total_population):
    plt.text(bar.get_width() + 500000, bar.get_y() + bar.get_height() / 2,
             f'{value / 1000000:.1f} млн', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

# Вывод статистики
print(f"Всего населения в федеральных округах: {sum(total_population)/1000000:.1f} млн")
print(f"Самый населенный округ: {district_names[total_population.index(max(total_population))]} - {max(total_population)/1000000:.1f} млн")
print(f"Наименее населенный округ: {district_names[total_population.index(min(total_population))]} - {min(total_population)/1000000:.1f} млн")
# График 2: Мужское население
plt.figure(figsize=(12, 8))
bars = plt.barh(district_names, male_population, color='lightblue', edgecolor='darkblue', alpha=0.8)
plt.xlabel('Количество мужчин', fontsize=12)
plt.title('Количество мужчин по федеральным округам', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)

# Добавляем подписи значений
for bar, value in zip(bars, male_population):
    plt.text(bar.get_width() + 300000, bar.get_y() + bar.get_height() / 2,
             f'{value / 1000000:.1f} млн', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

# Статистика по мужчинам
print(f"Всего мужчин в федеральных округах: {sum(male_population)/1000000:.1f} млн")
print(f"Доля мужчин от общего населения: {sum(male_population)/sum(total_population)*100:.1f}%")
# График 3: Женское население
plt.figure(figsize=(12, 8))
bars = plt.barh(district_names, female_population, color='lightpink', edgecolor='darkred', alpha=0.8)
plt.xlabel('Количество женщин', fontsize=12)
plt.title('Количество женщин по федеральным округам', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)

# Добавляем подписи значений
for bar, value in zip(bars, female_population):
    plt.text(bar.get_width() + 300000, bar.get_y() + bar.get_height() / 2,
             f'{value / 1000000:.1f} млн', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

# Статистика по женщинам
print(f"Всего женщин в федеральных округах: {sum(female_population)/1000000:.1f} млн")
print(f"Доля женщин от общего населения: {sum(female_population)/sum(total_population)*100:.1f}%")
print(f"Перевес женщин: {(sum(female_population)-sum(male_population))/1000000:.1f} млн человек")
# График 4: Круговая диаграмма соотношения полов
plt.figure(figsize=(10, 8))

labels = ['Мужчины', 'Женщины']
sizes = [male_total, female_total]
colors = ['lightblue', 'lightpink']
explode = (0.05, 0)  # Немного отделяем первый сегмент

wedges, texts, autotexts = plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                                   autopct='%1.1f%%', shadow=True, startangle=90,
                                   textprops={'fontsize': 12})

# Делаем подписи жирными
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')

plt.axis('equal')
plt.title('Соотношение мужчин и женщин в России', fontsize=14, fontweight='bold')

# Добавляем общую информацию внизу
total = male_total + female_total
plt.text(0, -1.5, f'Всего: {total/1000000:.1f} млн человек\n'
                   f'Мужчины: {male_total/1000000:.1f} млн\n'
                   f'Женщины: {female_total/1000000:.1f} млн\n'
                   f'Разница: {abs(male_total-female_total)/1000000:.1f} млн',
         ha='center', fontsize=11, bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
plt.show()
#КЕЙС 2
# Чтение данных из JSON файла
try:
    df = pd.read_json('2.json', encoding='utf-8')
    print("✅ Файл успешно загружен!")
    print(f"\nРазмер данных: {df.shape}")
    print(f"\nПервые 3 строки:")
    print(df.head(3))
    print(f"\nПоследние 3 строки:")
    print(df.tail(3))
except Exception as e:
    print(f"❌ Ошибка при загрузке файла: {e}")
    print("Создаю тестовые данные для демонстрации...")

    # Создаем тестовые данные
    years = list(range(2005, 2024))
    age_groups = [
        "от 20 до 24 лет",
        "от 25 до 29 лет",
        "от 30 до 34 лет",
        "от 35 до 39 лет",
        "от 40 до 44 лет",
        "от 45 до 49 лет",
        "от 50 до 54 лет",
        "от 55 до 59 лет",
        "от 60 до 64 лет",
        "65 лет и старше"
    ]

    # Создаем DataFrame с тестовыми данными
    np.random.seed(42)
    test_data = []
    for year in years:
        base = 20000 + (year - 2005) * 2000
        column_data = {}
        for i, group in enumerate(age_groups):
            # Реалистичная модель роста зарплат
            age_factor = 1 + i * 0.1  # Зарплата растет с возрастом
            exp_factor = 1 + (year - 2005) * 0.05  # Опыт с годами
            salary = base * age_factor * exp_factor * np.random.uniform(0.95, 1.05)
            column_data[group] = salary
        test_data.append(column_data)

    df = pd.DataFrame(test_data).T
    df.columns = years
    print("\nСозданы тестовые данные для демонстрации")
# Анализируем структуру загруженных данных
print("🔍 АНАЛИЗ СТРУКТУРЫ ДАННЫХ")
print("=" * 50)

if isinstance(df, pd.DataFrame):
    print(f"1. Тип данных: {type(df)}")
    print(f"2. Размер: {df.shape[0]} строк, {df.shape[1]} столбцов")
    print(f"3. Индекс: {df.index.name if df.index.name else 'Без названия'}")
    print(f"4. Типы данных колонок:")
    print(df.dtypes.head(10))

    # Проверяем первые несколько строк для понимания структуры
    print(f"\n5. Содержимое первых 5 строк:")
    for i in range(min(5, len(df))):
        print(f"   Строка {i}: {df.iloc[i].name if hasattr(df.iloc[i], 'name') else 'Без имени'}")
        if len(df.columns) > 0:
            sample_val = df.iloc[i, min(1, len(df.columns)-1)]
            print(f"     Пример значения: {sample_val}")

    # Проверяем наличие числовых данных
    print(f"\n6. Проверка числовых данных:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"   Найдено {len(numeric_cols)} числовых колонок")
        print(f"   Примеры: {list(numeric_cols[:3])}")
    else:
        print("   Числовые колонки не найдены, возможно данные в нестандартном формате")
else:
    print("Данные не в формате DataFrame")
print("🛠️ ПРЕДОБРАБОТКА ДАННЫХ")
print("=" * 50)

# Создаем копию данных для обработки
data_df = df.copy()

# Вариант 1: Если данные уже в правильном формате (возрастные группы как индекс)
if isinstance(df.index[0], str) and any(keyword in str(df.index[0]).lower() for keyword in ['от', 'лет', 'возраст']):
    print("✅ Данные уже в правильном формате (возрастные группы как индекс)")
    clean_df = df.astype(float)

# Вариант 2: Если данные в "длинном" формате
else:
    print("🔄 Преобразование данных в нужный формат...")

    # Ищем колонки с годами
    year_cols = []
    for col in df.columns:
        try:
            # Пробуем преобразовать название колонки в число (год)
            year = int(str(col).strip())
            if 1900 <= year <= 2100:
                year_cols.append(col)
        except:
            pass

    if len(year_cols) > 0:
        print(f"   Найдены годы: {year_cols}")
        # Если есть колонка с возрастными группами
        age_col = None
        for col in df.columns:
            if col not in year_cols and any(keyword in str(col).lower() for keyword in ['возраст', 'группа', 'age']):
                age_col = col
                break

        if age_col:
            print(f"   Найдена колонка с возрастными группами: {age_col}")
            clean_df = df.pivot_table(index=age_col, values=year_cols, aggfunc='mean')
        else:
            print("   Колонка с возрастными группами не найдена, использую первую колонку")
            clean_df = df.set_index(df.columns[0])
    else:
        print("   Годы не найдены, создаю тестовые данные...")
        # Создаем тестовые данные
        years = list(range(2005, 2024))
        age_groups = [
            "от 20 до 24 лет",
            "от 25 до 29 лет",
            "от 30 до 34 лет",
            "от 35 до 39 лет",
            "от 40 до 44 лет",
            "от 45 до 49 лет",
            "от 50 до 54 лет",
            "от 55 до 59 лет",
            "от 60 до 64 лет",
            "65 лет и старше"
        ]

        np.random.seed(42)
        data = []
        for age in age_groups:
            base_salary = np.random.randint(20000, 80000)
            row = [base_salary * (1 + 0.05 * (year - 2005)) * np.random.uniform(0.95, 1.05) for year in years]
            data.append(row)

        clean_df = pd.DataFrame(data, index=age_groups, columns=years)

print(f"\n✅ Итоговый DataFrame:")
print(f"   Размер: {clean_df.shape}")
print(f"   Возрастные группы: {len(clean_df)}")
print(f"   Годы: {list(clean_df.columns)}")

# Показываем первые строки
print(f"\nПервые 3 возрастные группы:")
print(clean_df.head(3))
print("📊 ГРАФИК 1: Динамика зарплат по возрастным группам")
print("=" * 50)

plt.figure(figsize=(14, 8))

# Ограничиваем количество групп для читаемости (первые 8)
groups_to_show = min(8, len(clean_df))
age_groups_display = clean_df.index[:groups_to_show]

for i, age_group in enumerate(age_groups_display):
    # Создаем цветовую палитру
    colors = plt.cm.tab10(np.linspace(0, 1, groups_to_show))
    plt.plot(clean_df.columns, clean_df.loc[age_group],
             marker='o', linewidth=2.5, markersize=6,
             label=age_group[:20] + ('...' if len(age_group) > 20 else ''),
             color=colors[i], alpha=0.8)

plt.xlabel('Год', fontsize=13, fontweight='bold')
plt.ylabel('Средняя заработная плата, руб.', fontsize=13, fontweight='bold')
plt.title('Динамика средней заработной платы по возрастным группам',
          fontsize=15, fontweight='bold', pad=20)

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
           fontsize=10, frameon=True, shadow=True)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()

# Добавляем аннотацию
last_year = clean_df.columns[-1]
first_year = clean_df.columns[0]
plt.text(0.02, 0.98, f'Период: {first_year}-{last_year} гг.\nГрупп показано: {groups_to_show} из {len(clean_df)}',
         transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.show()

# Выводим статистику по графикам
print(f"\n📊 СТАТИСТИКА ПО ГРАФИКУ 1:")
print(f"• Показано {groups_to_show} возрастных групп из {len(clean_df)}")
print(f"• Анализируемый период: {first_year}-{last_year} годы")
print(f"• Количество лет в анализе: {len(clean_df.columns)}")
print("📊 ГРАФИК 2: Зарплаты по возрастным группам (последний год)")
print("=" * 50)

plt.figure(figsize=(14, 8))

# Берем данные за последний год
latest_year = clean_df.columns[-1]
salaries_latest = clean_df[latest_year]

# Сортируем по убыванию
sorted_indices = salaries_latest.argsort()[::-1]
sorted_salaries = salaries_latest.iloc[sorted_indices]
sorted_groups = [clean_df.index[i] for i in sorted_indices]

# Создаем короткие названия для отображения
short_groups = []
for group in sorted_groups:
    # Извлекаем числа из названия группы
    import re
    numbers = re.findall(r'\d+', group)
    if len(numbers) >= 2:
        short_groups.append(f'{numbers[0]}-{numbers[1]} лет')
    elif len(numbers) == 1:
        if 'старше' in group.lower():
            short_groups.append(f'{numbers[0]}+ лет')
        else:
            short_groups.append(f'от {numbers[0]} лет')
    else:
        short_groups.append(group[:15] + ('...' if len(group) > 15 else ''))

# Создаем цветовую шкалу в зависимости от зарплаты
norm_salaries = (sorted_salaries - sorted_salaries.min()) / (sorted_salaries.max() - sorted_salaries.min())
colors = plt.cm.viridis(norm_salaries)

bars = plt.bar(short_groups, sorted_salaries, color=colors,
               edgecolor='black', linewidth=1.2, alpha=0.85)

plt.xlabel('Возрастные группы', fontsize=13, fontweight='bold')
plt.ylabel('Средняя заработная плата, руб.', fontsize=13, fontweight='bold')
plt.title(f'Средняя заработная плата по возрастным группам ({latest_year} год)',
          fontsize=15, fontweight='bold', pad=20)

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(True, alpha=0.3, axis='y', linestyle='--')

# Добавляем значения на столбцы
for bar, value in zip(bars, sorted_salaries):
    height = bar.get_height()
    # Форматируем число с пробелами для тысяч
    formatted_value = f'{int(value):,}'.replace(',', ' ')
    plt.text(bar.get_x() + bar.get_width()/2, height * 1.01,
             formatted_value, ha='center', va='bottom',
             fontsize=9, fontweight='bold', rotation=0)

# Добавляем горизонтальную линию среднего значения
mean_salary = sorted_salaries.mean()
plt.axhline(y=mean_salary, color='red', linestyle='--', linewidth=2, alpha=0.7,
            label=f'Среднее: {mean_salary:,.0f} руб.'.replace(',', ' '))

plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.show()

# Выводим статистику
print(f"\n📊 СТАТИСТИКА ПО ГРАФИКУ 2 ({latest_year} год):")
print(f"• Самая высокая зарплата: {sorted_groups[0]} - {sorted_salaries.iloc[0]:,.0f} руб.".replace(',', ' '))
print(f"• Самая низкая зарплата: {sorted_groups[-1]} - {sorted_salaries.iloc[-1]:,.0f} руб.".replace(',', ' '))
print(f"• Разница: {sorted_salaries.iloc[0] - sorted_salaries.iloc[-1]:,.0f} руб.".replace(',', ' '))
print(f"• Средняя зарплата: {mean_salary:,.0f} руб.".replace(',', ' '))
print(f"• Соотношение макс/мин: {sorted_salaries.iloc[0] / sorted_salaries.iloc[-1]:.1f} раз")
print("📊 ГРАФИК 3: Тепловая карта зарплат по годам и группам")
print("=" * 50)

plt.figure(figsize=(16, 10))

# Сортируем возрастные группы по возрасту (извлекаем первое число)
def extract_min_age(group_name):
    import re
    numbers = re.findall(r'\d+', str(group_name))
    if numbers:
        return int(numbers[0])
    return 100  # Для групп без чисел

# Сортируем группы по возрасту
sorted_age_indices = sorted(range(len(clean_df.index)),
                            key=lambda i: extract_min_age(clean_df.index[i]))
sorted_data = clean_df.iloc[sorted_age_indices]

# Создаем тепловую карту
im = plt.imshow(sorted_data.values, cmap='YlOrRd', aspect='auto',
                interpolation='nearest', vmin=sorted_data.values.min(),
                vmax=sorted_data.values.max())

# Настройка осей
plt.xticks(range(len(sorted_data.columns)),
           [str(year) for year in sorted_data.columns],
           rotation=45, fontsize=10)
plt.yticks(range(len(sorted_data.index)),
           [str(idx)[:20] + ('...' if len(str(idx)) > 20 else '')
            for idx in sorted_data.index],
           fontsize=10)

plt.xlabel('Год', fontsize=13, fontweight='bold')
plt.ylabel('Возрастная группа', fontsize=13, fontweight='bold')
plt.title('Тепловая карта заработных плат\nпо годам и возрастным группам',
          fontsize=15, fontweight='bold', pad=20)

# Добавляем цветовую шкалу
cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
cbar.set_label('Зарплата, руб.', fontsize=12, fontweight='bold')

# Добавляем значения в ячейки (только если данных не слишком много)
if len(sorted_data) <= 15 and len(sorted_data.columns) <= 15:
    for i in range(len(sorted_data)):
        for j in range(len(sorted_data.columns)):
            value = sorted_data.iloc[i, j]
            # Определяем цвет текста в зависимости от фона
            norm_value = (value - sorted_data.values.min()) / (sorted_data.values.max() - sorted_data.values.min())
            text_color = 'white' if norm_value > 0.6 else 'black'

            plt.text(j, i, f'{int(value/1000):.0f}K',
                     ha="center", va="center",
                     color=text_color, fontsize=8, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\n📊 СТАТИСТИКА ПО ТЕПЛОВОЙ КАРТЕ:")
print(f"• Всего возрастных групп: {len(sorted_data)}")
print(f"• Всего лет анализа: {len(sorted_data.columns)}")
print(f"• Диапазон зарплат: {sorted_data.values.min():,.0f} - {sorted_data.values.max():,.0f} руб.".replace(',', ' '))
print("📊 ГРАФИК 4: Рост зарплат с 2005 по 2023 год")
print("=" * 50)

plt.figure(figsize=(14, 8))

# Проверяем наличие данных за 2005 и 2023 годы
if 2005 in clean_df.columns and 2023 in clean_df.columns:
    # Вычисляем рост в процентах
    growth_rates = ((clean_df[2023] - clean_df[2005]) / clean_df[2005] * 100)

    # Убираем бесконечные значения и NaN
    growth_rates = growth_rates.replace([np.inf, -np.inf], np.nan).dropna()

    if len(growth_rates) > 0:
        # Сортируем по росту
        growth_sorted = growth_rates.sort_values(ascending=False)

        # Создаем короткие названия
        short_names = []
        for group in growth_sorted.index:
            import re
            numbers = re.findall(r'\d+', str(group))
            if len(numbers) >= 2:
                short_names.append(f'{numbers[0]}-{numbers[1]}')
            elif len(numbers) == 1:
                if 'старше' in str(group).lower():
                    short_names.append(f'{numbers[0]}+')
                else:
                    short_names.append(f'от {numbers[0]}')
            else:
                short_names.append(str(group)[:12] + ('...' if len(str(group)) > 12 else ''))

        # Определяем цвета в зависимости от роста
        colors = []
        for rate in growth_sorted.values:
            if rate > 250:
                colors.append('#27ae60')  # Темно-зеленый
            elif rate > 200:
                colors.append('#2ecc71')  # Зеленый
            elif rate > 150:
                colors.append('#f1c40f')  # Желтый
            elif rate > 100:
                colors.append('#e67e22')  # Оранжевый
            else:
                colors.append('#e74c3c')  # Красный

        # Создаем столбцы
        bars = plt.bar(short_names, growth_sorted.values, color=colors,
                      edgecolor='black', linewidth=1.2, alpha=0.85)

        plt.xlabel('Возрастные группы', fontsize=13, fontweight='bold')
        plt.ylabel('Рост зарплаты, %', fontsize=13, fontweight='bold')
        plt.title(f'Рост заработной платы с 2005 по 2023 год',
                  fontsize=15, fontweight='bold', pad=20)

        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.grid(True, alpha=0.3, axis='y', linestyle='--')

        # Добавляем линию среднего роста
        mean_growth = growth_sorted.mean()
        plt.axhline(y=mean_growth, color='blue', linestyle='--',
                    linewidth=2.5, alpha=0.8,
                    label=f'Средний рост: {mean_growth:.1f}%')

        # Добавляем значения на столбцы
        for bar, value in zip(bars, growth_sorted.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                    f'+{value:.0f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

        plt.legend(loc='upper right', fontsize=10)
        plt.tight_layout()
        plt.show()

        # Выводим статистику
        print(f"\n📊 СТАТИСТИКА РОСТА ЗАРПЛАТ (2005-2023):")
        print(f"• Средний рост по всем группам: {mean_growth:.1f}%")
        print(f"• Максимальный рост: {growth_sorted.index[0]} - +{growth_sorted.iloc[0]:.1f}%")
        print(f"• Минимальный рост: {growth_sorted.index[-1]} - +{growth_sorted.iloc[-1]:.1f}%")
        print(f"• Разброс роста: {growth_sorted.iloc[0] - growth_sorted.iloc[-1]:.1f}%")

        # Анализ по возрастным категориям
        young_groups = [i for i in growth_sorted.index if '20' in str(i) or '25' in str(i) or '30' in str(i)]
        middle_groups = [i for i in growth_sorted.index if '35' in str(i) or '40' in str(i) or '45' in str(i)]
        older_groups = [i for i in growth_sorted.index if '50' in str(i) or '55' in str(i) or '60' in str(i) or '65' in str(i)]

        if young_groups:
            young_avg = growth_sorted[young_groups].mean()
            print(f"• Средний рост молодых групп (20-34): {young_avg:.1f}%")
        if middle_groups:
            middle_avg = growth_sorted[middle_groups].mean()
            print(f"• Средний рост средних групп (35-49): {middle_avg:.1f}%")
        if older_groups:
            older_avg = growth_sorted[older_groups].mean()
            print(f"• Средний рост старших групп (50+): {older_avg:.1f}%")
    else:
        print("❌ Нет данных для расчета роста зарплат")
else:
    print(f"❌ Отсутствуют данные за 2005 и/или 2023 год")
    print(f"   Доступные годы: {list(clean_df.columns)}")
# КЕЙС 3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
import warnings
warnings.filterwarnings('ignore')

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-darkgrid')
print("✅ Библиотеки импортированы")
import os

# Проверяем, загружен ли файл
if not os.path.exists('3.xml'):
    print("❌ Файл 3.xml не найден!")
    print("Пожалуйста, загрузите файл используя меню слева")
else:
    print("✅ Файл 3.xml найден")

    # Парсинг XML
    try:
        tree = ET.parse('3.xml')
        root = tree.getroot()
        print("✅ XML файл успешно прочитан")

        # Извлекаем данные
        data = []
        for obj in root.findall('object'):
            row = {}
            for child in obj:
                text = child.text.strip() if child.text else None
                if text and text != '':
                    try:
                        # Для температуры делим на 10 (десятые доли)
                        if child.tag == 'Temp':
                            row[child.tag] = float(text) / 10.0
                        else:
                            row[child.tag] = float(text)
                    except ValueError:
                        row[child.tag] = text
                else:
                    row[child.tag] = None
            data.append(row)

        # Создаем DataFrame
        df = pd.DataFrame(data)
        df = df.replace('', np.nan)

        print(f"✅ Создан DataFrame с {len(df)} записями")

    except Exception as e:
        print(f"❌ Ошибка при чтении файла: {e}")
# Проверяем, создан ли DataFrame
if 'df' in locals() and len(df) > 0:
    print("="*60)
    print("СТРУКТУРА ДАННЫХ")
    print("="*60)
    print(f"Количество строк: {len(df)}")
    print(f"Количество столбцов: {len(df.columns)}")
    print("\nСтолбцы:")
    for col in df.columns:
        print(f"  - {col}")

    print(f"\n{'='*60}")
    print("ПЕРВЫЕ 5 СТРОК")
    print("="*60)
    print(df.head())

    print(f"\n{'='*60}")
    print("ИНФОРМАЦИЯ О ДАННЫХ")
    print("="*60)
    print(df.info())

    print(f"\n{'='*60}")
    print("ОСНОВНЫЕ СТАТИСТИКИ")
    print("="*60)
    print(df.describe())
else:
    print("❌ DataFrame не создан. Пожалуйста, сначала загрузите файл data.xml")
# Проверяем, создан ли DataFrame
if 'df' in locals() and len(df) > 0:
    print("📊 Создание графиков...")

    # Создаем графики
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Анализ атмосферных данных (2011 год)', fontsize=16, fontweight='bold')

    # 1. Температура по месяцам на разных уровнях давления
    ax1 = axes[0, 0]
    pressure_levels = [1111, 1000, 850, 500, 300]
    colors = plt.cm.viridis(np.linspace(0, 1, len(pressure_levels)))

    for level, color in zip(pressure_levels, colors):
        level_data = df[df['Pres'] == level].dropna(subset=['Temp', 'Month'])
        if not level_data.empty:
            monthly_avg = level_data.groupby('Month')['Temp'].mean()
            ax1.plot(monthly_avg.index, monthly_avg.values, 'o-',
                    linewidth=2, label=f'{level} гПа',
                    color=color, markersize=6, markerfacecolor='white')

    ax1.set_xlabel('Месяц', fontsize=12)
    ax1.set_ylabel('Температура (°C)', fontsize=12)
    ax1.set_title('Температура по месяцам на разных уровнях давления', fontsize=14)
    ax1.legend(title='Уровень давления', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(range(1, 7))
    ax1.set_xticklabels(['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн'])

    # 2. Высота vs Давление (март)
    ax2 = axes[0, 1]
    march_data = df[(df['Month'] == 3) & (df['Time'] == 0)].dropna(subset=['Hight', 'Pres'])

    if not march_data.empty:
        scatter = ax2.scatter(march_data['Hight'], march_data['Pres'],
                             c=march_data['Temp'], cmap='coolwarm',
                             s=100, alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Высота (м)', fontsize=12)
        ax2.set_ylabel('Давление (гПа)', fontsize=12)
        ax2.set_title('Зависимость высоты от давления (Март 2011, 00:00)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()
        plt.colorbar(scatter, ax=ax2).set_label('Температура (°C)', fontsize=12)
    else:
        ax2.text(0.5, 0.5, 'Нет данных за март 00:00',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=12)

    # 3. Сезонная динамика температуры на поверхности
    ax3 = axes[1, 0]
    surface_data = df[df['Pres'] == 1111].dropna(subset=['Temp', 'Month', 'Time'])

    if not surface_data.empty:
        time_0 = surface_data[surface_data['Time'] == 0].groupby('Month')['Temp'].mean()
        time_12 = surface_data[surface_data['Time'] == 12].groupby('Month')['Temp'].mean()

        months = range(1, 7)
        width = 0.35

        ax3.bar(np.array(months) - width/2, time_0, width, label='00:00',
                color='navy', alpha=0.7, edgecolor='black')
        ax3.bar(np.array(months) + width/2, time_12, width, label='12:00',
                color='coral', alpha=0.7, edgecolor='black')

        ax3.set_xlabel('Месяц', fontsize=12)
        ax3.set_ylabel('Температура (°C)', fontsize=12)
        ax3.set_title('Температура на поверхности по месяцам и времени суток', fontsize=14)
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_xticks(months)
        ax3.set_xticklabels(['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн'])
    else:
        ax3.text(0.5, 0.5, 'Нет данных для поверхности',
                 ha='center', va='center', transform=ax3.transAxes, fontsize=12)

    # 4. Модуль скорости ветра по высоте (апрель)
    ax4 = axes[1, 1]
    april_data = df[(df['Month'] == 4) & (df['Time'] == 0)].dropna(subset=['Modul', 'Hight'])

    if not april_data.empty:
        april_data = april_data.sort_values('Hight')
        ax4.plot(april_data['Modul'], april_data['Hight'], '^-',
                linewidth=2, markersize=8, color='green',
                markerfacecolor='lightgreen', markeredgecolor='darkgreen')

        ax4.fill_betweenx(april_data['Hight'], 0, april_data['Modul'],
                         alpha=0.2, color='green')

        ax4.set_xlabel('Модуль скорости ветра (м/с)', fontsize=12)
        ax4.set_ylabel('Высота (м)', fontsize=12)
        ax4.set_title('Скорость ветра по высоте (Апрель 2011, 00:00)', fontsize=14)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Нет данных за апрель 00:00',
                 ha='center', va='center', transform=ax4.transAxes, fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    print("✅ Графики созданы успешно")
else:
    print("❌ Невозможно создать графики: DataFrame не найден")
# Проверяем, создан ли DataFrame
if 'df' in locals() and len(df) > 0:
    print("📊 СТАТИСТИЧЕСКИЙ АНАЛИЗ")
    print("="*60)

    # Основная информация
    print(f"Период данных: {int(df['Year'].iloc[0])} год, месяцы {int(df['Month'].min())}-{int(df['Month'].max())}")
    print(f"Количество измерений: {len(df):,}")
    print(f"Уровни давления: {len(df['Pres'].unique())}")

    # Диапазоны значений
    numeric_columns = ['Temp', 'Hight', 'Modul']
    for col in numeric_columns:
        if col in df.columns and df[col].notna().any():
            print(f"Диапазон {col}: от {df[col].min():.1f} до {df[col].max():.1f}")

    # Средние температуры по месяцам на поверхности
    print(f"\n{'='*60}")
    print("СРЕДНЯЯ ТЕМПЕРАТУРА ПО МЕСЯЦАМ (ПОВЕРХНОСТЬ)")
    print("="*60)

    surface_data = df[df['Pres'] == 1111].dropna(subset=['Temp', 'Month'])
    if not surface_data.empty:
        surface_temps = surface_data.groupby('Month')['Temp'].mean()
        for month, temp in surface_temps.items():
            print(f"Месяц {month}: {temp:.1f}°C")
    else:
        print("Нет данных для поверхности")

    # Анализ по месяцам
    print(f"\n{'='*60}")
    print("ОБЩАЯ СТАТИСТИКА ПО МЕСЯЦАМ")
    print("="*60)

    monthly_stats = df.groupby('Month').agg({
        'Temp': ['mean', 'min', 'max'],
        'Hight': 'max',
        'Modul': 'mean'
    })

    print(monthly_stats.round(1))

    # Анализ по уровням давления
    print(f"\n{'='*60}")
    print("ТЕМПЕРАТУРА ПО УРОВНЯМ ДАВЛЕНИЯ")
    print("="*60)

    pressure_stats = df.groupby('Pres')['Temp'].agg(['mean', 'min', 'max']).round(1)
    print(pressure_stats)

else:
    print("❌ Невозможно выполнить статистический анализ: DataFrame не найден")
# Проверяем, создан ли DataFrame
if 'df' in locals() and len(df) > 0:
    print("📈 Дополнительная визуализация...")

    # График 1: Тепловая карта температуры
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    # Готовим данные для тепловой карты
    heatmap_data = pd.pivot_table(
        df.dropna(subset=['Temp', 'Month', 'Pres']),
        values='Temp',
        index='Pres',
        columns='Month',
        aggfunc='mean'
    )

    # Сортируем по давлению (убыванию)
    heatmap_data = heatmap_data.sort_index(ascending=False)

    # Создаем тепловую карту
    im = plt.imshow(heatmap_data, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
    plt.colorbar(im, label='Температура (°C)')
    plt.title('Температура: давление × месяц')
    plt.xlabel('Месяц')
    plt.ylabel('Давление (гПа)')
    plt.xticks(range(6), ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн'])
    plt.yticks(range(len(heatmap_data)), [str(int(x)) for x in heatmap_data.index])

    # График 2: Распределение температур
    plt.subplot(1, 2, 2)
    all_temps = df['Temp'].dropna()

    plt.hist(all_temps, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    plt.axvline(x=all_temps.mean(), color='red', linestyle='--',
               linewidth=2, label=f'Среднее: {all_temps.mean():.1f}°C')
    plt.axvline(x=all_temps.median(), color='green', linestyle='--',
               linewidth=2, label=f'Медиана: {all_temps.median():.1f}°C')

    plt.xlabel('Температура (°C)', fontsize=12)
    plt.ylabel('Частота', fontsize=12)
    plt.title('Распределение температур', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("✅ Дополнительные графики созданы")
else:
    print("❌ Невозможно создать дополнительные графики: DataFrame не найден")
# КЕЙС 4
# Импорт необходимых библиотек
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import warnings
import io

warnings.filterwarnings('ignore')

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✅ Библиотеки импортированы")
# Загружаем данные из строки
df = pd.read_csv("4.csv")
print(f"✅ Загружено {len(df)} записей")
print(f"📊 Количество студентов: {df['student_id'].nunique()}")
print(f"📊 Количество вопросов: {df['question_number'].nunique()}")

print("\nПервые 15 строк данных:")
print(df.head(15))
# Добавляем столбец с правильностью ответа
df['is_correct'] = (df['answer'] == df['correct_answer']).astype(int)

print("="*80)
print("БАЗОВЫЙ АНАЛИЗ ДАННЫХ")
print("="*80)

print(f"\n📊 Общая статистика:")
print(f"   Количество студентов: {df['student_id'].nunique()}")
print(f"   Количество вопросов: {df['question_number'].nunique()}")
print(f"   Всего ответов: {len(df)}")
print(f"   Правильных ответов: {df['is_correct'].sum()} ({df['is_correct'].mean()*100:.1f}%)")

# Средний балл каждого студента
student_scores = df.groupby('student_id')['is_correct'].mean().reset_index()
student_scores.columns = ['student_id', 'average_score']

# Общий средний балл группы
overall_average = student_scores['average_score'].mean()

# Статистика по баллам
score_stats = {
    'Средний балл группы': overall_average,
    'Максимальный балл': student_scores['average_score'].max(),
    'Минимальный балл': student_scores['average_score'].min(),
    'Стандартное отклонение': student_scores['average_score'].std(),
    'Медиана': student_scores['average_score'].median(),
    'Коэффициент вариации': (student_scores['average_score'].std() / overall_average * 100)
}

print(f"\n📈 Статистика по баллам студентов:")
for k, v in score_stats.items():
    if k == 'Коэффициент вариации':
        print(f"   {k}: {v:.1f}%")
    else:
        print(f"   {k}: {v:.3f}")

print(f"\n🏆 Результаты студентов:")
for _, row in student_scores.iterrows():
    print(f"   Студент {row['student_id']}: {row['average_score']:.1%} правильных ответов ({int(row['average_score']*10)}/10)")
print("\n" + "="*80)
print("АНАЛИЗ СЛОЖНОСТИ ВОПРОСОВ")
print("="*80)

# Процент правильных ответов по вопросам
question_difficulty = df.groupby('question_number')['is_correct'].agg([
    'mean', 'count', 'std'
]).reset_index()
question_difficulty.columns = ['question_number', 'correct_rate', 'total_answers', 'std_dev']

# Классификация сложности
def classify_difficulty(rate):
    if rate < 0.4:
        return 'Очень сложный'
    elif rate < 0.6:
        return 'Сложный'
    elif rate < 0.8:
        return 'Средний'
    else:
        return 'Лёгкий'

question_difficulty['difficulty'] = question_difficulty['correct_rate'].apply(classify_difficulty)
question_difficulty_sorted = question_difficulty.sort_values('correct_rate', ascending=True)

print("\n📋 Сложность вопросов (от самых сложных):")
for _, row in question_difficulty_sorted.iterrows():
    correct_count = int(row['correct_rate'] * row['total_answers'])
    print(f"   Вопрос {row['question_number']}: {correct_count}/{row['total_answers']} правильных ({row['correct_rate']:.0%}) - {row['difficulty']}")

# Анализ распределения сложности
difficulty_counts = question_difficulty['difficulty'].value_counts()
print(f"\n📊 Распределение вопросов по сложности:")
for diff, count in difficulty_counts.items():
    print(f"   {diff}: {count} вопросов")

# Корреляция номера вопроса и сложности
correlation = question_difficulty['question_number'].corr(question_difficulty['correct_rate'])
print(f"\n📈 Корреляция номера вопроса и сложности: {correlation:.3f}")
print("\n" + "="*80)
print("АНАЛИЗ ПО ТИПАМ ЗАДАНИЙ")
print("="*80)

# Определение типов заданий
type_mapping = {
    1: 'Теория', 2: 'Теория', 3: 'Практика', 4: 'Практика',
    5: 'Анализ', 6: 'Анализ', 7: 'Расчёт', 8: 'Расчёт',
    9: 'Логика', 10: 'Логика'
}
df['question_type'] = df['question_number'].map(type_mapping)

# Анализ по типам
type_analysis = df.groupby('question_type')['is_correct'].agg([
    'mean', 'count', 'std', 'sem'
]).round(3)
type_analysis.columns = ['avg_correct_rate', 'num_answers', 'std_dev', 'std_error']
type_analysis['num_questions'] = type_analysis['num_answers'] / df['student_id'].nunique()

# Добавляем категорию успешности
def classify_success(rate):
    if rate >= 0.8:
        return 'Высокий'
    elif rate >= 0.6:
        return 'Средний'
    else:
        return 'Низкий'

type_analysis['success_category'] = type_analysis['avg_correct_rate'].apply(classify_success)

print("\n📊 Результаты по типам заданий:")
for type_name, row in type_analysis.iterrows():
    total_questions = int(row['num_questions'])
    correct_count = int(row['avg_correct_rate'] * row['num_answers'])
    print(f"\n   {type_name} ({total_questions} вопросов):")
    print(f"      Правильных ответов: {correct_count}/{int(row['num_answers'])} ({row['avg_correct_rate']:.0%})")
    print(f"      Стандартное отклонение: {row['std_dev']:.3f}")
    print(f"      Категория успешности: {row['success_category']}")
print("\n" + "="*80)
print("ПОДРОБНЫЙ АНАЛИЗ")
print("="*80)

# Матрица ответов
pivot_matrix = df.pivot_table(
    index='student_id',
    columns='question_number',
    values='is_correct',
    aggfunc='first'
).fillna(0)

print("\n📊 Матрица ответов студентов (1=правильно, 0=неправильно):")
print("   Строки - студенты, столбцы - вопросы")
print(pivot_matrix)

# Детальный анализ каждого студента
print("\n📋 Подробные результаты по студентам:")
for student_id in df['student_id'].unique():
    student_data = df[df['student_id'] == student_id]
    correct_answers = student_data['is_correct'].sum()
    total_questions = len(student_data)
    percentage = correct_answers / total_questions * 100

    # Анализ по типам вопросов для каждого студента
    type_performance = student_data.groupby('question_type')['is_correct'].mean()

    print(f"\n   Студент {student_id}: {correct_answers}/{total_questions} ({percentage:.0f}%)")
    for question_type, performance in type_performance.items():
        type_correct = int(performance * (total_questions / len(type_mapping)))
        print(f"      {question_type}: {type_correct}/2 ({performance:.0%})")

# Детальный анализ каждого вопроса
print("\n📋 Подробные результаты по вопросам:")
for question_num in sorted(df['question_number'].unique()):
    question_data = df[df['question_number'] == question_num]
    correct_count = question_data['is_correct'].sum()
    total_students = len(question_data)

    # Самые частые ошибки
    wrong_answers = question_data[question_data['is_correct'] == 0]
    if len(wrong_answers) > 0:
        common_wrong = wrong_answers['answer'].value_counts().head(2)
        common_wrong_str = ", ".join([f"{ans} ({count})" for ans, count in common_wrong.items()])
    else:
        common_wrong_str = "нет"

    print(f"   Вопрос {question_num} ({type_mapping[question_num]}):")
    print(f"      Правильных: {correct_count}/{total_students} ({correct_count/total_students:.0%})")
    print(f"      Частые ошибки: {common_wrong_str}")
print("\n📊 Создание графиков...")

# Создаем фигуру с несколькими графиками
fig = plt.figure(figsize=(16, 12))

# 1. Средние баллы студентов
ax1 = plt.subplot(2, 2, 1)
bars = ax1.bar(student_scores['student_id'], student_scores['average_score'] * 100,
              color=['red' if x < 0.6 else 'orange' if x < 0.8 else 'green'
                    for x in student_scores['average_score']])

ax1.set_title('Средние баллы студентов', fontsize=14, fontweight='bold')
ax1.set_xlabel('ID студента')
ax1.set_ylabel('Процент правильных ответов (%)')
ax1.set_xticks(student_scores['student_id'])
ax1.axhline(y=overall_average*100, color='blue', linestyle='--',
           label=f'Среднее по группе: {overall_average:.0%}')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Добавляем значения на столбцы
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height + 1,
            f'{height:.0f}%', ha='center', va='bottom', fontsize=10)

# 2. Сложность вопросов
ax2 = plt.subplot(2, 2, 2)
bars2 = ax2.bar(question_difficulty_sorted['question_number'],
               question_difficulty_sorted['correct_rate'] * 100,
               color=['red' if x == 'Очень сложный' else
                     'orange' if x == 'Сложный' else
                     'yellow' if x == 'Средний' else 'green'
                     for x in question_difficulty_sorted['difficulty']])

ax2.set_title('Сложность вопросов', fontsize=14, fontweight='bold')
ax2.set_xlabel('Номер вопроса')
ax2.set_ylabel('Процент правильных ответов (%)')
ax2.set_xticks(question_difficulty_sorted['question_number'])
ax2.grid(axis='y', alpha=0.3)

# Добавляем значения на столбцы
for bar, rate in zip(bars2, question_difficulty_sorted['correct_rate']):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + 1,
            f'{rate:.0%}', ha='center', va='bottom', fontsize=10)

# 3. Результаты по типам заданий
ax3 = plt.subplot(2, 2, 3)
type_bars = ax3.bar(range(len(type_analysis)),
                   type_analysis['avg_correct_rate'] * 100,
                   color=['red' if x == 'Низкий' else
                         'orange' if x == 'Средний' else 'green'
                         for x in type_analysis['success_category']])

ax3.set_title('Результаты по типам заданий', fontsize=14, fontweight='bold')
ax3.set_xlabel('Тип задания')
ax3.set_ylabel('Процент правильных (%)')
ax3.set_xticks(range(len(type_analysis)))
ax3.set_xticklabels(type_analysis.index, rotation=45, ha='right')
ax3.grid(axis='y', alpha=0.3)

# Добавляем значения на столбцы
for i, (bar, rate) in enumerate(zip(type_bars, type_analysis['avg_correct_rate'])):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, height + 1,
            f'{rate:.0%}', ha='center', va='bottom', fontsize=10)

# 4. Heatmap успеваемости
ax4 = plt.subplot(2, 2, 4)
heatmap_data = pivot_matrix.values
im = ax4.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

ax4.set_title('Матрица успеваемости', fontsize=14, fontweight='bold')
ax4.set_xlabel('Номер вопроса')
ax4.set_ylabel('ID студента')
ax4.set_xticks(range(len(pivot_matrix.columns)))
ax4.set_xticklabels(pivot_matrix.columns)
ax4.set_yticks(range(len(pivot_matrix.index)))
ax4.set_yticklabels(pivot_matrix.index)

# Добавляем значения в ячейки
for i in range(len(pivot_matrix.index)):
    for j in range(len(pivot_matrix.columns)):
        value = heatmap_data[i, j]
        color = 'black' if value == 1 else 'white'
        ax4.text(j, i, '✓' if value == 1 else '✗',
                ha='center', va='center', color=color, fontsize=12, fontweight='bold')

plt.colorbar(im, ax=ax4, label='Правильность (1=да, 0=нет)')
plt.tight_layout()
plt.show()
print("\n" + "="*80)
print("ДЕТАЛЬНЫЙ АНАЛИЗ И ВЫВОДЫ")
print("="*80)

# 1. Самые сложные вопросы
print("\n🔴 САМЫЕ СЛОЖНЫЕ ВОПРОСЫ:")
top_hard = question_difficulty_sorted.head(3)
for _, row in top_hard.iterrows():
    correct_count = int(row['correct_rate'] * row['total_answers'])
    print(f"   Вопрос {row['question_number']} ({type_mapping[row['question_number']]}):")
    print(f"      {correct_count}/{row['total_answers']} правильных ({row['correct_rate']:.0%})")

    # Анализ типичных ошибок
    wrong_data = df[(df['question_number'] == row['question_number']) & (df['is_correct'] == 0)]
    if len(wrong_data) > 0:
        common_wrong = wrong_data['answer'].value_counts()
        print(f"      Типичные ошибки: {', '.join([f'{ans} ({count})' for ans, count in common_wrong.items()])}")

# 2. Самые легкие вопросы
print("\n🟢 САМЫЕ ЛЕГКИЕ ВОПРОСЫ:")
top_easy = question_difficulty_sorted.tail(3).iloc[::-1]
for _, row in top_easy.iterrows():
    correct_count = int(row['correct_rate'] * row['total_answers'])
    print(f"   Вопрос {row['question_number']} ({type_mapping[row['question_number']]}):")
    print(f"      {correct_count}/{row['total_answers']} правильных ({row['correct_rate']:.0%})")

# 3. Лучшие и худшие студенты
print(f"\n🏆 ЛУЧШИЕ СТУДЕНТЫ:")
best_students = student_scores.nlargest(2, 'average_score')
for _, row in best_students.iterrows():
    correct_answers = int(row['average_score'] * 10)
    print(f"   Студент {row['student_id']}: {correct_answers}/10 ({row['average_score']:.0%})")

print(f"\n📉 СТУДЕНТЫ, ТРЕБУЮЩИЕ ВНИМАНИЯ:")
worst_students = student_scores.nsmallest(2, 'average_score')
for _, row in worst_students.iterrows():
    correct_answers = int(row['average_score'] * 10)
    print(f"   Студент {row['student_id']}: {correct_answers}/10 ({row['average_score']:.0%})")

# 4. Классификация студентов
print("\n📈 КЛАССИФИКАЦИЯ СТУДЕНТОВ:")

def classify_student(score):
    if score >= 0.9:
        return 'Отличник (A)'
    elif score >= 0.8:
        return 'Хорошист (B)'
    elif score >= 0.7:
        return 'Удовлетворительно (C)'
    elif score >= 0.6:
        return 'Слабо (D)'
    else:
        return 'Неудовлетворительно (F)'

student_scores['grade'] = student_scores['average_score'].apply(classify_student)
grade_counts = student_scores['grade'].value_counts()

for grade, count in grade_counts.items():
    percentage = count / len(student_scores) * 100
    print(f"   {grade}: {count} студентов ({percentage:.0f}%)")

# 5. Общая оценка теста
print("\n📊 ОБЩАЯ ОЦЕНКА ТЕСТА:")
print(f"   Средний балл группы: {overall_average:.1%}")
print(f"   Надежность теста (α Кронбаха): {question_difficulty['correct_rate'].std() / overall_average:.3f}")
print(f"   Дифференцирующая способность: {student_scores['average_score'].std():.3f}")

if overall_average >= 0.8:
    test_quality = "Отличный"
elif overall_average >= 0.7:
    test_quality = "Хороший"
elif overall_average >= 0.6:
    test_quality = "Удовлетворительный"
else:
    test_quality = "Слабый"

print(f"   Общая оценка теста: {test_quality}")

# 6. Рекомендации
print("\n" + "="*80)
print("РЕКОМЕНДАЦИИ")
print("="*80)

print("\n🎯 Для преподавателя:")
print("   1. Проанализировать вопросы с низкой успеваемостью")
print("   2. Рассмотреть возможность переформулирования сложных вопросов")
print("   3. Организовать дополнительные занятия по темам с низкой успеваемостью")

print("\n🎯 Для студентов:")
print("   1. Обратить внимание на вопросы, вызвавшие наибольшие затруднения")
print("   2. Проработать темы, связанные с типами вопросов, где процент правильных низкий")
print("   3. Использовать матрицу успеваемости для выявления слабых мест")

print("\n📈 Общий вывод:")
if overall_average >= 0.8:
    print("   ✅ Группа показала хорошие результаты, большинство студентов справились с тестом.")
elif overall_average >= 0.6:
    print("   ⚠️  Группа показала удовлетворительные результаты, есть над чем работать.")
else:
    print("   ❌ Группа показала слабые результаты, требуется дополнительное обучение.")

# 7. Таблица успеваемости
print("\n📋 СВОДНАЯ ТАБЛИЦА УСПЕВАЕМОСТИ:")
summary_df = pd.DataFrame({
    'Студент': student_scores['student_id'],
    'Балл (%)': (student_scores['average_score'] * 100).round(1),
    'Правильно': (student_scores['average_score'] * 10).astype(int),
    'Всего': 10,
    'Оценка': student_scores['grade']
})
print(summary_df.to_string(index=False))
# КЕЙС 5
# Импорт всех необходимых библиотек
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import io
import random  # Добавляем импорт random

warnings.filterwarnings('ignore')

# Настройка стиля
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
print("✅ Библиотеки импортированы")
def generate_test_data(num_students=50, num_questions=20, random_seed=42):
    """Генерация тестовых данных для анализа"""
    np.random.seed(random_seed)
    random.seed(random_seed)  # Теперь random определен

    # Типы вопросов
    question_types = {
        '1-5': 'multiple_choice_A',
        '6-10': 'true_false',
        '11-15': 'multiple_choice_B',
        '16-20': 'short_answer'
    }

    # Варианты ответов для разных типов вопросов
    correct_answers = {}
    student_answers = []

    # Генерация правильных ответов для каждого вопроса
    for q in range(1, num_questions + 1):
        if q <= 5:  # multiple choice A-D
            correct_answers[q] = random.choice(['A', 'B', 'C', 'D'])
        elif q <= 10:  # true/false
            correct_answers[q] = random.choice(['True', 'False'])
        else:  # short answer (простой текст)
            correct_answers[q] = f"Answer{q}"

    # Генерация ответов студентов
    for student_id in range(1, num_students + 1):
        # Уровень знаний студента (нормальное распределение)
        student_knowledge = np.random.normal(0.7, 0.2)
        student_knowledge = max(0.1, min(0.95, student_knowledge))  # ограничиваем

        # Факторы, влияющие на успеваемость
        motivation = np.random.normal(0.8, 0.15)  # мотивация
        test_anxiety = np.random.normal(0.3, 0.2)  # тревожность на тесте

        for question_id in range(1, num_questions + 1):
            correct_answer = correct_answers[question_id]

            # Вероятность правильного ответа с учетом факторов
            base_probability = student_knowledge * (1 - test_anxiety * 0.3) * motivation

            # Корректируем вероятность для разных типов вопросов
            if question_id <= 5:
                # Multiple choice - выше вероятность угадывания
                guess_probability = 0.25
                effective_prob = base_probability + (1 - base_probability) * guess_probability
                if random.random() < effective_prob:
                    student_answer = correct_answer
                else:
                    # Неправильный ответ, но с учетом вариантов
                    wrong_options = [opt for opt in ['A', 'B', 'C', 'D'] if opt != correct_answer]
                    student_answer = random.choice(wrong_options)

            elif question_id <= 10:
                # True/False - 50% вероятность угадывания
                guess_probability = 0.5
                effective_prob = base_probability + (1 - base_probability) * guess_probability
                if random.random() < effective_prob:
                    student_answer = correct_answer
                else:
                    student_answer = 'True' if correct_answer == 'False' else 'False'

            else:
                # Short answer - низкая вероятность угадывания
                if random.random() < base_probability:
                    student_answer = correct_answer
                else:
                    # Разные типы ошибок
                    error_types = [
                        f"Wrong{question_id}",
                        f"Alternative{question_id}",
                        f"Answer{question_id-1}",
                        f"Response{question_id}",
                        "Не знаю",
                        "Нет ответа"
                    ]
                    student_answer = random.choice(error_types)

            student_answers.append({
                'student_id': student_id,
                'question_id': question_id,
                'student_answer': student_answer,
                'correct_answer': correct_answer,
                'student_knowledge': student_knowledge,
                'motivation': motivation,
                'test_anxiety': test_anxiety
            })

    return pd.DataFrame(student_answers), correct_answers

# Генерация данных
df, correct_answers = generate_test_data(num_students=50, num_questions=20)

print("="*80)
print("🎯 ГЕНЕРАЦИЯ ТЕСТОВЫХ ДАННЫХ")
print("="*80)
print(f"✅ Сгенерировано {len(df)} записей")
print(f"📊 Количество студентов: {df['student_id'].nunique()}")
print(f"📊 Количество вопросов: {df['question_id'].nunique()}")
print(f"📊 Типы вопросов: Multiple Choice A (1-5), True/False (6-10), Multiple Choice B (11-15), Short Answer (16-20)")

# Сохранение данных
df.to_csv('test_results_extended.csv', index=False, encoding='utf-8')
print("\n💾 Данные сохранены в файл: test_results_extended.csv")
print("="*80)
print("📊 БАЗОВЫЙ АНАЛИЗ ДАННЫХ")
print("="*80)

# Добавляем колонку с правильностью ответа
df['is_correct'] = (df['student_answer'] == df['correct_answer']).astype(int)

# Общая статистика
total_students = df['student_id'].nunique()
total_questions = df['question_id'].nunique()
total_answers = len(df)
correct_answers_count = df['is_correct'].sum()
overall_accuracy = correct_answers_count / total_answers

print(f"\n📈 ОБЩАЯ СТАТИСТИКА:")
print(f"   Всего студентов: {total_students}")
print(f"   Всего вопросов: {total_questions}")
print(f"   Всего ответов: {total_answers}")
print(f"   Правильных ответов: {correct_answers_count} ({overall_accuracy*100:.1f}%)")

# Статистика по типам вопросов
def get_question_type(question_id):
    if question_id <= 5:
        return 'Multiple Choice A'
    elif question_id <= 10:
        return 'True/False'
    elif question_id <= 15:
        return 'Multiple Choice B'
    else:
        return 'Short Answer'

df['question_type'] = df['question_id'].apply(get_question_type)

type_stats = df.groupby('question_type')['is_correct'].agg([
    'mean', 'count', 'std', lambda x: (x == 1).sum()
]).reset_index()
type_stats.columns = ['question_type', 'accuracy', 'total_answers', 'std_dev', 'correct_count']
type_stats['num_questions'] = type_stats['total_answers'] / total_students

print(f"\n📊 СТАТИСТИКА ПО ТИПАМ ВОПРОСОВ:")
for _, row in type_stats.iterrows():
    print(f"   {row['question_type']}:")
    print(f"      Вопросов: {int(row['num_questions'])}")
    print(f"      Правильных ответов: {row['correct_count']}/{int(row['total_answers'])} ({row['accuracy']*100:.1f}%)")
    print(f"      Стандартное отклонение: {row['std_dev']:.3f}")
print("="*80)
print("👨‍🎓 АНАЛИЗ СТУДЕНТОВ")
print("="*80)

# Статистика по студентам
student_stats = df.groupby('student
