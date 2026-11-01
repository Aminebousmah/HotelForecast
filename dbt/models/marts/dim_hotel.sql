select hotel_id, brand, region, capacity
from {{ source('raw', 'hotels') }}
