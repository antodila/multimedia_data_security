# Multimedia Data Security - Competition Rules (2025)

## What is the competition about?
The Multimedia Data Security competition is essentially a group activity that allows applying already learned concepts in a fun way. The objectives are two:
• Applying an embedding strategy that is both robust and unperceivable
• Attacking the watermarked information of other groups while preserving image quality

## Scheduling
**Deadlines:**
- Wednesday, 06th October 11.59 PM - GROUP FORMATION
- Sunday, 27th October 11.59 PM - EMBEDDING, DETECTION, ATTACK and ROC CODE SUBMISSION

**Events:**
- Monday, 3th November 8.30 AM- 12.30 PM - COMPETITION
- Wednesday 5st November 11.30 AM- 13.30 PM - GROUPS PRESENTATIONS

## Group Formation
All participating students are divided into groups of 3/4 people. Each group must select:
• a spokesperson, that will take care of the communication with the teaching assistant
• a nickname that only your group will know. Group nicknames must not contain spaces/symbols/capital letters.

The object of the mail MUST BE: [CTM 2025]- groupname.
This must be done by the following deadline: 06th October at 11.59 PM.

After the deadline for registration, you'll receive an e-mail providing:
• A randomly generated watermark which will be strictly associated with your group
• A randomly generated password that you will need to access the website of the competition

## Running Code Submission
By 27th October at 11.59PM you will be asked to deliver a working code. This is just to avoid you having problems the day of the competition, and have the backup of a working solution. You are asked to deliver also the ROC-curve code so that we can check how you set the threshold.

**Meaning of Fine-tuning**: adjust parameters and embedding locations. Other things are forbidden.

## Competition
The competition will take place on 3th November, please be ready to start at 8.30 AM.

**Defense phase → 8.30 AM- 10 AM**
During the defense phase, you will have to:
1. Download the challenge's images (three 512 × 512 grayscale images)
2. Download the WPSNR code (if needed, as it will be the same code provided during laboratories)
3. Use your embedding strategy to insert your watermark in each of the three images
4. Upload the embedded images to the website. You can do this multiple times, each time you do it a score will be assigned to you based on the quality of the embedding
5. Upload the code of your detection strategy (other groups will use it during the attack phase)

**Attack phase → 10.15 AM- 12.30 AM**
The second phase of the competition consists in targeting the other groups and attacking their images following these steps:
1. download other groups images from the list that will appear on the website
2. performs some image processing attack(s) on the images watermarked by other groups to remove the watermark while keeping image quality as high as possible
3. uploads the attacked images on the website, indicating which group was attacked

**NOTE**: using the original image to localize the watermark is NOT allowed.

## Verification phase
After the competition you are required to send a log of your attacks (use excel). The required structure is:
| Image | Group | WPSNR | Attack(s) with parameters |
|-------|-------|-------|-------------------------|
| lena | attackedGroup1 | 37 | JPEG QF=90, Median 5×5 |

## Embedding
The code for the embedding MUST follow this structure:
```python
def embedding(input1, input2):
    '''
    YOUR CODE
    '''
    return output1
```
• input1 corresponds to the string of the name of the original image
• input2 corresponds to the string of the name of the watermark
• output1: is the watermarked image

**NOTE**: Your code must NOT print anything, neither open interactive pages or windows or similar.

## Detection
The detection function must be a single file named detection_groupname.py, no external functions are allowed (except those of the WPSNR).

The function will make use of the WPSNR code seen during the laboratory sessions.

**YOU MUST NOT HAVE RELATIVE PATH OR EXTERNAL SCRIPTS CALLED IN THIS FUNCTION**

```python
def detection(input1, input2, input3):
    '''
    YOUR CODE
    '''
    return output1, output2
```
• input1 corresponds to the string of the name of the original image
• input2 corresponds to the string of the name of the watermarked image
• input3 corresponds to the string of the name of the attacked image
• output1: if the attacked image contains the watermark it is equal to 1, otherwise it is equal to 0
• output2 corresponds to the WPSNR value between the watermarked and the attacked image.

**An attack is considered successful if:**
• the similarity is below the threshold τ (i.e., output1 = 0)
• the WPSNR ≥ 35 [dB] (i.e., output2 ≥ 35).

**Final remark**: your code must complete the detection within 5 seconds and must not open any pop-up windows or print anything on the screen.

## Attack
Permitted attacks for the competition are:
• AWGN
• Blurring
• Sharpening
• JPEG Compression
• Resizing
• Median filtering

The code CAN follow this structure:
```python
def attacks(input1, attack_name, param_array):
    '''
    YOUR CODE
    '''
    return output1
```

## Naming convention
• Embedded images: groupA_imageName.bmp
• Downloaded images: groupB_imageName.bmp
• Attacked images: groupA_groupB_imageName.bmp

## Scoring Information
**EMBEDDING QUALITY:**
| WPSNR | POINTS |
|-------|--------|
| 35 ≤ WPSNR < 50 | 1 |
| 50 ≤ WPSNR < 54 | 2 |
| 54 ≤ WPSNR < 58 | 3 |
| 58 ≤ WPSNR < 62 | 4 |
| 62 ≤ WPSNR < 66 | 5 |
| WPSNR ≥ 66 | 6 |

**ROBUSTNESS:**
| WPSNR | POINTS |
|-------|--------|
| 35 ≤ WPSNR < 38 | 6 |
| 38 ≤ WPSNR < 41 | 5 |
| 41 ≤ WPSNR < 44 | 4 |
| 44 ≤ WPSNR < 47 | 3 |
| 47 ≤ WPSNR < 50 | 2 |
| 50 ≤ WPSNR < 53 | 1 |
| WPSNR ≥ 53 | 0 |

**ACTIVITY:**
| % OF GROUPS ATTACKED | POINTS |
|----------------------|--------|
| >30% | 2 |
| >60% | 4 |
| >90% | 6 |

**QUALITY:**
| ATTACKED IMAGES WITH WPSNR >AVG WPSNR | POINTS |
|----------------------------------------|--------|
| 1-5 | 1 |
| 6-10 | 2 |
| 11-15 | 3 |
| > 15 | 4 |

**BONUS:** 2 extra points if you successfully attacked a group that no one else attacked or if you were not attacked by anyone.

## Penalties
A 2-point penalty in the competition results will be applied for each transgression of the discussed rules, including:
• You miss a deadline
• You have to change one of the files uploaded to the website after the conclusion of the defense phase
• The detection function successfully finds a watermark in non-watermarked images, in several unrelated images, or destroyed images (WPSNR ≤ 25db)
• Your detection code opens pop-up windows or prints on the screen
• Your detection code takes more than 5 seconds to run
