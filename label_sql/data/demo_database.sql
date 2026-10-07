-- Optional disposable demo database setup. Not run by this project automatically.
CREATE TABLE departments (
    department_id integer PRIMARY KEY,
    department_name text NOT NULL
);
CREATE TABLE employees (
    employee_id integer PRIMARY KEY,
    full_name text NOT NULL,
    department_id integer REFERENCES departments(department_id),
    salary numeric(12,2) NOT NULL
);
CREATE TABLE cases (
    case_id integer PRIMARY KEY,
    employee_id integer NOT NULL REFERENCES employees(employee_id),
    status text NOT NULL,
    overpayment_amount numeric(12,2) NOT NULL,
    opened_on date NOT NULL
);
INSERT INTO departments VALUES (1, 'Finance'), (2, 'Engineering');
INSERT INTO employees VALUES (100, 'Anita', 1, 65000), (101, 'Ravi', 2, 80000);
INSERT INTO cases VALUES (200, 100, 'Open', 7500, '2026-01-15'), (201, 101, 'Closed', 2000, '2026-02-10');
