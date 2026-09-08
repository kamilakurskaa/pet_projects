--Топ-3 авторов с наибольшим MAU в ноябре
WITH author_stats AS(
SELECT main_author_name,
EXTRACT(month from msk_business_dt_str) AS month
COUNT(DISTINCT puid) AS mau
FROM bookmate.audition au
JOIN bookmate.content  c ON au.main_content_id = c.main_content_id,
JOIN bookmate.author a ON c.main_author_id = a.main_author_id
WHERE EXTRACT(month from msk_business_dt_str) == 11
GROUP BY EXTRACT(month from msk_business_dt_str), main_author_name
)

SELECT main_author_name, mau
FROM author_stats
ORDER BY mau DESC
LIMIT 3;

--Топ-3 произведений с наибольшим MAU в ноябре
WITH books_stats AS (
SELECT main_content_name,
published_topic_title_list,
main_author_name,
COUNT(DISTINCT puid) AS mau
FROM bookmate.audition au
JOIN bookmate.content c ON au.main_content_id = c.main_content_id
JOIN bookmate.author a ON c.main_author_id = a.main_author_id
WHERE EXTRACT(month from msk_business_dt_str) = 11
GROUP BY main_content_name, published_topic_title_list, main_author_name
)
SELECT main_content_name,
published_topic_title_list,
main_author_name,
mau
FROM books_stats
ORDER BY mau DESC
LIMIT 3;

--Retention Rate 
WITH cohort AS (
SELECT DISTINCT puid
FROM bookmate.audition
WHERE msk_business_dt_str == '2024-12-02'
),
activity AS (
SELECT DISTINCT puid,
(CAST(a.msk_business_dt_str AS date) - CAST('2024-12-02' AS date)) AS day_since_install
FROM bookmate.audition a
JOIN cohort c ON a.puid = c.puid
WHERE msk_business_dt_str >= '2024-12-02'
)
SELECT day_since_install,
COUNT(DISTINCT puid) AS retained_users,
COUNT(DISTINCT puid) * 100.0/ COUNT(SELECT puid FROM cohort)
FROM activity
GROUP BY day_since_install
ORDER BY day_since_install;

--LTV для пользователей в Москве и Санкт-Петербурге
WITH active_users AS(
puid,
usage_geo_id_name AS city,
EXTRACT (month from msk_business_dt_str) AS month
FROM bookmate.audition au
JOIN bookmate.geo g ON au.usage_geo_id = g.usage_geo_id
WHERE city IN ('Москва', 'Санкт-Петербург')
),
payments AS(
SELECT puid, city, COUNT(DISTINCT month) AS months_paid
FROM active_users
GROUP BY puid, city
),
city_revenue AS(
SELECT city, 
COUNT(DISTINCT puid) AS total_users,
SUM(months_paid) * 399.0 AS total_revenue
FROM payments
GROUP BY city
)
SELECT city, total_users,
ROUND(total_revenue / total_users, 2) AS ltv
FROM city_revenue
ORDER BY city;

--Расчёт средней выручки прослушанного часа
WITH stats AS(
SELECT EXTRACT(month from msk_business_dt_str) AS month.
COUNT(DISTINCT puid) AS mau,
SUM(hours) AS hours,
FROM bookmate.audition
GROUP BY EXTRACT(month from msk_business_dt_str)
)
SELECT month,
mau,
ROUND(hours,  2) AS hours,
ROUND(mau * 399.0 / NULLIF(hours, 0), 2) AS avg_hour_rev
FROM stats