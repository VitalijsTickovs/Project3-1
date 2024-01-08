# Weight file instructions:
## Abbreviations
- standard: 
    
    (34\*3, 50), (50,40), (40,30), 

    (30,40), (40,50), (34\*3, 50)

- prototype 2 (p2):

    (34\*3, 50), (50,40), (40,35), (35,30)

    (30,35), (35,40), (40,50), (34\*3, 50)

- prototype 3 (p3):

    (34\*3, 90), (90,70), (70,50), (50,30)

    (30,50), (50,70), (70,90), (90, 34\*3)

    Failure. Reached only 0.06348 + slower trainign time

## Files
1) new.pth

    Currently p2
2) w_0576.pth
    
    weights for standard model (0.0576 accuracy)

3) w_p2_0550.pth

    for p2 model (0.550 accuracy)

4) p3.pth

    for failed p3 model (0.06348)
