# Лабораторная работа 1

**Предметная область:** задержки авиарейсов — *US DOT Flight Delays* (5 819 079 строк, 2015 – 2020 гг.)
https://www.kaggle.com/datasets/usdot/flight-delays?spm=a2ty_o01.29997173.0.0.50abc9211ZZTsW

**Аппаратно‑ПО:** PostgreSQL 14.18 (Homebrew) · macOS ARM (M1) · 16 GB RAM · NVMe SSD

---

## 0 Подготовка данных

```bash
# 0.1 Скачиваем и распаковываем датасет
kaggle datasets download -d usdot/flight-delays -p ~/data
unzip ~/data/flight-delays.zip -d ~/data/flights

# 0.2 Создаём БД и подключаемся
createdb flight_delays
psql flight_delays
```

```sql
-- 0.3 Минимальная схема (достаточно для всех пунктов работы)
CREATE TABLE flights (
    flight_id            BIGSERIAL PRIMARY KEY,
    year                 SMALLINT,
    month                SMALLINT,
    day                  SMALLINT,
    flight_date          DATE GENERATED ALWAYS AS (make_date(year,month,day)) STORED,
    airline              CHAR(2),
    flight_number        INTEGER,
    origin_airport       CHAR(3),
    destination_airport  CHAR(3),
    scheduled_departure  SMALLINT,
    departure_time       SMALLINT,
    departure_delay      SMALLINT,
    scheduled_arrival    SMALLINT,
    arrival_time         SMALLINT,
    arrival_delay        SMALLINT,
    distance             INTEGER,
    diverted             BOOLEAN,
    cancelled            BOOLEAN,
    route TEXT GENERATED ALWAYS AS ((origin_airport||'-'||destination_airport)) STORED
);

-- 0.4 Загружаем данные (≈ 5 минут на SSD)
\copy flights(year,month,day,airline,flight_number,origin_airport,destination_airport,
              scheduled_departure,departure_time,departure_delay,scheduled_arrival,arrival_time,
              arrival_delay,distance,diverted,cancelled)
      FROM '~/data/flights/flights.csv' CSV HEADER;
```

> **Итого:** в таблице `flights` — 5 819 079 строк ✅

---

## 1.1 Типы индексов и их эффективность

### 1.1.1 Теория

| Тип        | Как устроен                                | Когда использовать                                   |
| ---------- | ------------------------------------------ | ---------------------------------------------------- |
| **B‑tree** | Сбалансированное дерево поиск = O(log n)   | равенство и диапазоны на селективных столбцах        |
| **BRIN**   | хранилище min/max для каждых *N* страниц   | огромные таблицы с монотонным ключом (дата, ID)      |
| **GIN**    | инвертированный список значений/фрагментов | массивы, JSONB, полнотекст, `%LIKE%` через trgm/bigm |

### 1.1.2 Замер до индексов

```sql
-- Q1 фильтр по компании и месяцу
EXPLAIN (ANALYZE,BUFFERS)
SELECT * FROM flights
 WHERE airline='AA' AND flight_date BETWEEN '2015-07-01' AND '2015-07-31';
-- ≈ 0.586 с   (Parallel Seq Scan)

-- Q2 фильтр по дате
EXPLAIN (ANALYZE,BUFFERS)
SELECT * FROM flights WHERE flight_date >= '2020-01-01';
-- ≈ 0.251 с   (Parallel Seq Scan)

-- Q3 fuzzy‑маршрут
EXPLAIN (ANALYZE,BUFFERS)
SELECT * FROM flights WHERE route LIKE '%JFK-LAX%';
-- ≈ 0.229 с   (Parallel Seq Scan)
```

### 1.1.3 Создание индексов

```sql
-- B‑tree по (airline, flight_date)
CREATE INDEX flights_airline_date_btree
          ON flights (airline, flight_date);

-- BRIN по дате
CREATE INDEX flights_date_brin
          ON flights USING brin (flight_date) WITH (pages_per_range = 64);

-- GIN + по route
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- если ещё не
CREATE INDEX flights_route_trgm_gin
          ON flights USING gin (route);
```

### 1.1.4 Сравнение после индексации

| Запрос                                                                                                   | Время         | Индекс |
| -------------------------------------------------------------------------------------------------------- | ------------- | ------ |
| AA + июль‑2015                                                                                           | **0.125 с**   | B‑tree |
| flight\_date ≥ 2020‑01‑01                                                                                | **0.00013 с** | BRIN   |
| `%JFK-LAX%`                                                                                              | **0.153 с**   | GIN     |


---

```sql
-- Q1 ▸ AA + июль-2015  → B-tree (Bitmap Index Scan)
EXPLAIN (ANALYZE,BUFFERS)
SELECT *
  FROM flights
 WHERE airline = 'AA'
   AND flight_date BETWEEN '2015-07-01' AND '2015-07-31';
-- 0.125 с  (Bitmap Index Scan → Bitmap Heap)

-- Q2 ▸ flight_date ≥ 2020-01-01  → BRIN
EXPLAIN (ANALYZE,BUFFERS)
SELECT *
  FROM flights
 WHERE flight_date >= '2020-01-01';
-- 0.00013 с  (Bitmap Heap Scan + BRIN recheck)

-- Q3 ▸ «%JFK-LAX%»  → селективность ещё высока ⇒ остаётся Seq Scan
EXPLAIN (ANALYZE,BUFFERS)
SELECT *
  FROM flights
 WHERE route LIKE '%JFK-LAX%';
-- 0.153 с  (Parallel Seq Scan)
```

> **Итог:** ускорение до 50× при весом в 40 MB (B‑tree) и 48 kB (BRIN).

---

## 1.2 Транзакции и уровни изоляции

### Теория

PostgreSQL реализует MVCC: каждая транзакция читает «снимок» данных.
Уровни изоляции отличаются тем, **когда** этот снимок обновляется:

* **READ COMMITTED** – перед каждым SQL‑оператором → допускает non‑repeatable/phantom reads.
* **REPEATABLE READ** – один снимок на всю транзакцию → фантомов нет, но возможен lost‑update.
* **SERIALIZABLE** – добавляет проверку конфликтов; видит мир как последовательность, откатывая «нарушителей».

### 1.2.1 Non‑repeatable read (RC vs RR)

```sql
-- А (RC)
BEGIN;
SELECT AVG(departure_delay) FROM flights WHERE flight_date='2015-07-01';

-- B
BEGIN; UPDATE flights SET departure_delay=departure_delay+10
 WHERE flight_date='2015-07-01'; COMMIT;

-- А
SELECT AVG(departure_delay) ...  -- выросло
COMMIT;
```

В `REPEATABLE READ` та же последовательность сохраняет первоначальное среднее.

### 1.2.2 Phantom read

```sql
-- таблица-мини
CREATE TABLE daily_flights AS
SELECT flight_id, flight_date, airline
  FROM flights WHERE flight_date='2015-07-02' AND airline='AA' LIMIT 100;

-- B (RC)
BEGIN; SELECT COUNT(*) FROM daily_flights;  -- 100

-- A
BEGIN; INSERT INTO daily_flights SELECT ... OFFSET 100 LIMIT 1; COMMIT;

-- B
SELECT COUNT(*) FROM daily_flights;  -- 101 (фантом)
COMMIT;
```

Под `REPEATABLE READ` обе выборки возвращают 100.

### 1.2.3 Lost update (RC vs SER)

```sql
CREATE TABLE seat_inventory(flight_id BIGINT PRIMARY KEY, seats_left INT);
INSERT INTO seat_inventory VALUES (4288379,2);

-- A (RC)
BEGIN; UPDATE seat_inventory SET seats_left=seats_left-1 WHERE flight_id=4288379;
-- без COMMIT

-- B
BEGIN; UPDATE seat_inventory SET seats_left=seats_left-1 WHERE flight_id=4288379; COMMIT;

-- A
COMMIT;   -- в RC проходит, seats_left потеряно
```

Под `SERIALIZABLE` второй `COMMIT` падает: *could not serialize access due to concurrent update*.

---

## 1.3 Расширения pg\_trgm · pg\_bigm · pgcrypto

### 1.3.1 Substring‑поиск: trgm vs bigm

```sql
CREATE EXTENSION IF NOT EXISTS pg_bigm;
CREATE INDEX flights_route_bigm_gin
          ON flights USING gin (route gin_bigm_ops);
```

| Индекс   | Exec‑time `%JFK-LAX%` | Примечание                               |
| -------- | --------------------: | ---------------------------------------- |
| GIN‑trgm |            **164 мс** | Bitmap Index Scan                        |
| GIN‑bigm |            **335 мс** | Planner оставил Seq Scan (селективность) |

*trgm* поддерживает `similarity()` и fuzzy‑операторы:

```sql
SELECT route, similarity(route,'JFK-LXA')
  FROM flights WHERE route % 'JFK-LXA' ORDER BY 2 DESC LIMIT 5;
```

### 1.3.2 Безопасность: pgcrypto

```sql
-- Шифруем tail_number
ALTER TABLE flights ADD COLUMN tail_enc BYTEA;
UPDATE flights SET tail_enc = pgp_sym_encrypt(tail_number,'super‑secret');

-- Проверка
SELECT pgp_sym_decrypt(tail_enc,'super‑secret') FROM flights LIMIT 3;

-- BCrypt‑пароли
CREATE TABLE app_users(user_id BIGSERIAL PRIMARY KEY,
                       login TEXT UNIQUE,
                       pwd_hash TEXT);
INSERT INTO app_users(login,pwd_hash)
VALUES ('alice', crypt('Pa$$w0rd', gen_salt('bf')));
```
