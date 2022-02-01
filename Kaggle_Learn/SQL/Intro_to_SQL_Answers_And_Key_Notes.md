### Key points about SQL

- Structured Query Language to access data from Relational databases
- Case Insensitive: <br>
    - SQL keywords are by default set to **case insensitive** 
    - The names of the tables and columns specification are set to case insensitive on the SQL database server, 
    - However, it can be enabled and disabled by configuring the settings in SQL.

### Common data types in SQL (sqlite) and most important in other SQL tools

- NULL. The value is a NULL value.
- INTEGER. The value is a signed integer, stored in 0, 1, 2, 3, 4, 6, or 8 bytes depending on the magnitude of the value.
- REAL. The value is a floating point value, stored as an 8-byte IEEE floating point number.
- TEXT. The value is a text string, stored using the database encoding (UTF-8, UTF-16BE or UTF-16LE).
- BLOB. The value is a blob of data, stored exactly as it was input (image, audio, etc.,) <br>
([source](https://www.sqlite.org/datatype3.html))

### Types of Queries

1. Query Type: <br>
- `SELECT ... FROM ... WHERE` 

```
    SELECT DISTINCT country
    FROM `bigquery-public-data.openaq.global_air_quality`
    WHERE unit = 'ppm'
```

2. Query Type: <br>
- `SELECT ... FROM ... WHERE ... LIKE` 

```
    SELECT DISTINCT country
    FROM `bigquery-public-data.some_db.some_table`
    WHERE unit LIKE '%bigquery%'
```

3. Query Type: <br>
- `SELECT ... FROM ... GROUP BY ... HAVING ... `

```
SELECT parent, COUNT(1) AS NumPosts
FROM `bigquery-public-data.hacker_news.comments`
GROUP BY parent
HAVING COUNT(1) > 10
```

4. Query Type: <br>
- `SELECT ... FROM ... WHERE ... GROUP BY ... HAVING ... ORDER BY DESC|ASC`

```
    SELECT indicator_code, indicator_name, COUNT(indicator_code) as num_rows
    FROM `bigquery-public-data.world_bank_intl_education.international_education`
    WHERE year = 2016
    GROUP BY indicator_code, indicator_name
    HAVING num_rows >= 175
    ORDER BY num_rows DESC
```

5. Query Type: (exploring SQL EXTRACT FROM datetime variables) <br>
- `SELECT (EXTRACT DAYOFWEEK|MONTH|YEAR|DAYOFYEAR FROM time_stamp_column AS some_name FROM ... GROUP BY ... ORDER BY ... `

```
SELECT COUNT(consecutive_number) AS num_accidents, 
       EXTRACT(DAYOFWEEK FROM timestamp_of_crash) AS day_of_week
FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`
GROUP BY day_of_week
ORDER BY num_accidents DESC
```

Source: [Bigquery docyumentation of available time_stamp related keywords](https://cloud.google.com/bigquery/docs/reference/standard-sql/timestamp_functions) 

6. Query Type: (CTE - Common Table Expression) <br>
- `WITH TEMP AS (CTE) SELECT some_column FROM TEMP GROUP BY ... ORDER BY ...`

```
 WITH time AS 
 (
     SELECT DATE(block_timestamp) AS trans_date
     FROM `bigquery-public-data.crypto_bitcoin.transactions`
 )
 SELECT COUNT(1) AS transactions,
        trans_date
 FROM time
 GROUP BY trans_date
 ORDER BY trans_date
```

7. Query Type: (JOIN): <br>
- `SELECT table1.col1, table1.col2, table2.col1 FROM table1 INNER JOIN table2 ON table1.PRIMARY_KEY = table2.FOREIGN_KEY` <br>
- A primary key is a column or a set of columns in a table whose values uniquely identify a row in the table
- A foreign key is a column or a set of columns in a table whose values correspond to the values of the primary key in another table

```
SELECT a.owner_user_id AS user_id, COUNT(a.id) AS number_of_answers
FROM `bigquery-public-data.stackoverflow.posts_questions` AS q
INNER JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
    ON q.id = a.parent_id
WHERE q.tags LIKE '%bigquery%'
GROUP BY user_id
```

8. Query Type (`CREATE`, `DROP`, `INSERT`)

```
DROP TABLE IF EXISTS marks_data;

CREATE TABLE marks_data (
                        grade_class integer,
                        marks integer integer,
                        student_id integer PRIMARY_KEY,
                        names text
                        );
                        
-- If importing from a CSV                        
-- SELECT IMPORT("path/to/grade_marks.csv", "CSV", "marks_data");

INSERT INTO marks_data VALUES(‘12’,78,'S56','Senthil')
```

9. Query Type (`sub_query`): <br>
- Sub querying in FROM: `SELECT A, B FROM (select  tabl2.col1 AS A, tabl2.col2 AS B FROM table2)`
- Sub quering in SELECT: `SELECT account, level, (SELECT AVG(level) FROM Players) AS avg_level FROM Players`
- so many other varieties ... 

```
SELECT grade_class, student_id, marks
FROM 
(
SELECT grade_class, marks, student_id, RANK() OVER(
    PARTITION BY grade_class
    ORDER BY marks
    ) marks_rank
FROM marks_data_2
)
WHERE marks_rank=3
```
