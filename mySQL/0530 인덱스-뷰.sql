use sakila;

-- replace는:없으면 만들고, 있으면 수정해라(5ver이후부터 됨)
CREATE OR REPLACE VIEW v_customer AS
SELECT CONCAT(a.last_name, a.first_name) AS customer_name,
       postal_code, district, phone, location, address
FROM customer a
JOIN address b ON a.address_id = b.address_id;

-- 가상의테이블 
SELECT * FROM v_customer
WHERE customer_name LIKE '%smith%';
