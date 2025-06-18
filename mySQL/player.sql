
-- player.sql: MySQL-compatible SQL script to create and populate a 'player' table

CREATE TABLE IF NOT EXISTS player (
    player_id INT AUTO_INCREMENT PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    position VARCHAR(30),
    team VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO player (first_name, last_name, position, team)
VALUES 
    ('Lionel', 'Messi', 'Forward', 'Inter Miami'),
    ('Cristiano', 'Ronaldo', 'Forward', 'Al Nassr'),
    ('Kevin', 'De Bruyne', 'Midfielder', 'Manchester City'),
    ('Virgil', 'van Dijk', 'Defender', 'Liverpool'),
    ('Manuel', 'Neuer', 'Goalkeeper', 'Bayern Munich');
