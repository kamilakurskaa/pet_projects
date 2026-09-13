--Получение общих данных
select currency_code,
sum(p.revenue) as total_revenue,
count(order_id) as total_orders,
avg(revenue) as avg_revenue_per_order,
count(distinct user_id) as total_users
from purchases p
group by currency_code
order by total_revenue desc;

--Распределение выручки в разрезе устройств
select device_type_canonical,
sum(p.revenue) as total_revenue,
count(order_id) as total_orders,
avg(revenue) as avg_revenue_per_order,
ROUND(sum(revenue)::numeric / (select sum(revenue)::numeric from purchases where currency_code='rub'), 3) as revenue_share
from purchases p 
where p.currency_code = 'rub'
group by device_type_canonical
order by revenue_share desc;

--Распределение выручки в разрезе типа мероприятия
select e.event_type_main,
sum(revenue) as total_revenue,
count(order_id) as total_orders,
avg(revenue) as avg_revenue_per_order,
count(distinct e.event_name_code) as total_event_name,
avg(p.tickets_count) as avg_tickets,
sum(revenue)/sum(tickets_count) as avg_ticket_revenue,
sum(revenue) / (select sum(revenue) from purchases where currency_code='rub') as revenue_share
from purchases p 
join events e on p.event_id = e.event_id 
where p.currency_code = 'rub'
group by e.event_type_main
order by total_orders desc;

--Динамика изменений значений
select date_trunc('week', created_dt_msk)::date as week,
sum(revenue) as total_revenue,
count(order_id) as total_orders,
count(distinct user_id) as total_users,
avg(revenue) as revenue_per_order
from purchases
where currency_code='rub'
group by week
order by week;

--Выделение топ-сегментов
select r.region_name,
sum(p.revenue) as total_revenue,
count(order_id) as total_orders,
count(distinct p.user_id) as total_users,
sum(p.tickets_count) as total_tickets,
sum(p.revenue) / sum(p.tickets_count) as one_ticket_cost
from purchases p
join events e on p.event_id = e.event_id
join city c on e.city_id = c.city_id
join regions r on c.region_id = r.region_id
where currency_code='rub'
group by r.region_name
order by total_revenue desc
limit 7;