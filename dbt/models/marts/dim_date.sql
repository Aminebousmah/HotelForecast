select date, dow, month, doy, is_public_holiday, school_holiday_share
from {{ source('raw', 'dates') }}
