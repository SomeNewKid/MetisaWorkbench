# Storage inspection checklist

1. Check motor temperature first. If it is above 35°C, stop and
   tell the user to let the motor cool to 35°C or below.
2. Once the temperature check passes, check battery charge.
   The acceptable storage range is 40% to 60%, inclusive.
   If the charge is below 40%, tell the user to charge the battery
   into that range. If it is above 60%, tell the user to reduce
   the charge into that range.
3. If both the motor temperature and battery charge checks pass,
   use the `storage_available.star` script to determine if storage
   is available for the named robot.
3. Report that the robot is ready for storage if both checks pass,
   and storage is available.

If a required measurement is unavailable, or the script cannot be executed, 
report that the inspection could not be completed.