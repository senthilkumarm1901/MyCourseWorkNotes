## hackerrank SQL Questions and Answers 
(This file holds my answers and notes from solely for learning and not for sharing answer)

### Sample list of Different Questions

1. Query distinct of one column with condition on another column <br>
```SQL
SELECT DISTINCT NAME FROM CITY WHERE COUNTRYCODE = 'JPN';
```

2. Query all rows with an even value for a column
```SQL
SELECT DISTINCT CITY FROM STATION WHERE MOD(ID,2) = 0;
```

3. Finding difference between two columns (but after doing `distinct_count` and `count`) <br>

:x: Failed Attempt in Hackerrank:
```
WITH TEMP AS (SELECT COUNT(CITY) AS count1, COUNT(DISTINCT CITY) AS count2 FROM STATION) SELECT count1 - count2 FROM TEMP;
```

☑️ What worked: 
```SQL
SELECT count1 - count2 FROM (SELECT COUNT(CITY) AS count1, COUNT(DISTINCT CITY) AS count2 FROM STATION);
```

4. Find the Lowest and Highest of a column and display the corresponding values from another column
- Two select statements are needed and hence a union is required

```SQL
SELECT * FROM (
    SELECT CITY , MAX(LENGTH(CITY)) AS LEN_CITY FROM STATION GROUP BY CITY ORDER BY LEN_CITY DESC LIMIT 1
) UNION 
SELECT * FROM (
    SELECT CITY , MIN(LENGTH(CITY)) AS LEN_CITY FROM STATION GROUP BY CITY ORDER BY LEN_CITY ASC LIMIT 1
);
```

```
amo 3
some big name of a city 21
```

5. Query a TEXT column for a particular pattern <br>
- Use `LIKE` for finding pattern in text column; `OR`/ `AND` when there are multiple conditions

```SQL
SELECT CITY 
FROM STATION 
WHERE CITY LIKE 'A%' OR 
      CITY LIKE 'E%' OR 
      CITY LIKE 'I%' OR
      CITY LIKE 'O%' OR
      CITY LIKE 'U%';
```

```SQL
SELECT DISTINCT CITY 
FROM STATION 
WHERE NOT (CITY LIKE 'A%' OR 
      CITY LIKE 'E%' OR 
      CITY LIKE 'I%' OR
      CITY LIKE 'O%' OR
      CITY LIKE 'U%');
```

- SUB QUERY if an `OR` and an `AND` needs to be used 

```SQL
SELECT DISTINCT CITY 
FROM (
SELECT DISTINCT CITY
FROM STATION
WHERE CITY LIKE 'A%' OR 
      CITY LIKE 'E%' OR 
      CITY LIKE 'I%' OR
      CITY LIKE 'O%' OR
      CITY LIKE 'U%') 
WHERE CITY LIKE '%a' OR
      CITY LIKE '%e' OR
      CITY like '%i' OR
      CITY like '%o' OR
      CITY like '%u';
```

```SQL
SELECT DISTINCT CITY
FROM STATION
WHERE NOT (CITY LIKE 'A%' OR
           CITY LIKE 'E%' OR
           CITY LIKE 'I%' OR
           CITY LIKE 'O%' OR
           CITY LIKE 'U%'
          )
UNION
SELECT DISTINCT CITY 
FROM STATION
WHERE NOT (CITY LIKE '%a' OR
           CITY LIKE '%e' OR
           CITY LIKE '%i' OR
           CITY LIKE '%o' OR
           CITY LIKE '%u');
```

6. SQL `String` functions <br>
- Doing operations based on string function
- Used `RIGHT` function in the below example <br>
- Other `string` functions in SQL; Look [here](https://www.w3schools.com/sql/sql_ref_sqlserver.asp)


```SQL
SELECT Name 
FROM (
SELECT RIGHT(Name,3) as Lastchars, Name, ID, Marks
FROM STUDENTS
)
WHERE Marks > 75
ORDER BY Lastchars, ID; 
```

:x: What did not work?
```SQL
SELECT Name 
FROM (
    SELECT Name, ID, Marks, RANK() OVER(
        PARTITION BY ID
        ORDER BY RIGHT(Name,3)
    ) name_rank
    FROM STUDENTS
    WHERE Marks > 75
)
ORDER BY name_rank;
```
