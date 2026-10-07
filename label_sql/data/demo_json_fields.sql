-- Optional extension AFTER demo_database.sql, in a disposable database.
-- Never executed automatically by this project.
ALTER TABLE employees ADD COLUMN profile jsonb;
UPDATE employees SET profile = '{"address":{"city":"Chennai"},"bonus":12500,"joined_on":"2025-01-15","active":true}'::jsonb WHERE employee_id = 100;
UPDATE employees SET profile = '{"address":{"city":"Salem"},"bonus":9000,"joined_on":"2024-04-01","active":false}'::jsonb WHERE employee_id = 101;
