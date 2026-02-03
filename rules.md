
Benchmarks/Competitions
Datasets
Competition Logo
NTIRE 2026 Light Field Image Super-Resolution Challenge - Track 2 Efficiency
Organized by: fyzhang
Current Phase Ends: March 5, 2026 at 7:00 PM EST
Current server time: January 31, 2026 at 1:11 PM EST
Docker image: codalab/codalab-legacy:py39 
NTIRE 2026: Light Field Image Super-Resolution Challenge - Track 2 Efficiency
Training Set
This challenge follows the training set in the paper DistgSSR, and uses the EPFL, HCInew, HCIold, INRIA and STFgantry datasets which totally consist of 144 scenes for training. All the LF images in the training set have an angular resolution of 9x9. Both HR LF images and their LR versions (produced by bicubic downsampling) can be downloaded via Baidu Drive (key:7nzy) or OneDrive. The participants can use these HR LF images as groundtruths, and use the BasicLFSR toolbox to train their models.


Validation Set
We collect a new validation set consisting of 16 synthetic scenes rendered by the 3DS MAX software and 16 real-world images captured by a Lytro ILLUM camera. We downsampled original LF images in the validation set by a factor of 4, and provide LR LF images with an angular resolution of 5x5. The participants can download the validation set to evaluate the performance of their developed models by submitting their super-resolved LF images to the Codabench server.


Test Set
We collect a new test set consisting of 16 synthetic scenes rendered by the 3DS MAX software and 16 real-world images captured by a Lytro ILLUM camera. Only 4× bicubically downsampled LR LF images with an angular resolution of 5x5 will be provided. The participants are required to apply their models to the released LR LF images and submit their 4× super-resolved LF images to the Codabench platform for final ranking. It should be noted that the images in both the validation and the test sets (even the LR versions) cannot be used for training.


Chasuite
Competitions v1.6
Chahub
Chagrade
About
About
Github
Privacy and Terms
API Docs
CodaBench
Join us on Github for contact & bug reports

Questions about the platform? See our Docs for more information.

1.22
Benchmarks/Competitions
Datasets
Competition Logo
NTIRE 2026 Light Field Image Super-Resolution Challenge - Track 2 Efficiency
Organized by: fyzhang
Current Phase Ends: March 5, 2026 at 7:00 PM EST
Current server time: January 31, 2026 at 1:11 PM EST
Docker image: codalab/codalab-legacy:py39 
NTIRE 2026: Light Field Image Super-Resolution Challenge
Important Dates
2026-01-20: Release of train data (input and output) and validation data (inputs only)
2026-01-21: Validation server online
2026-03-10: Final test data release (inputs only)
2026-03-17: Test output results submission deadline
2026-03-17: Fact sheets and code/executable submission deadline
2026-03-19: Preliminary test results release to the participants
2026-03-24: Paper submission deadline for entries from the challenge
2026-06(TBU): NTIRE workshop and challenges, results and award ceremony (CVPR 2026)

Introduction
With recent advances in camera manufacturing, light field (LF) imaging technology has become increasingly popular and is commonly used in various applications such as mobile phones, biological microscopy, VR/AR, etc. Since both intensity and directions of light rays are recorded by LF cameras, the resolution of LF images can be enhanced by using these additional angular information. LF image super-resolution (SR), also known as LF spatial SR, aims at reconstructing high-resolution (HR) LF images from their low-resolution (LR) counterparts.


Jointly with the NTIRE workshop, we organize a challenge for LF community to focus on enhancing the spatial resolution of LF images, and aspire to highlight the specific challenges and research problems faced by LF image SR. This challenge provides an opportunity for researchers to work together to share their knowledge and insights, advance the algorithm performance, and promote the development of LF image SR.


Challenge Description
The NTIRE 2026: New Trends in Image Restoration and Enhancement workshop will be held in June, 2026 in conjunction with CVPR 2026.


The objective of this challenge is to reconstruct HR LF images from their LR counterparts.
During the model development phase, the HR training set and the LR validation set will be released. The participants should train their models on the training set and submit their super-resolved validation images to the Codabench server for validation.
During the test phase, the test set will be released, which includes LR LF images only. Challenge participants should apply their trained models to the LR test images to generate super-resolved test images. These super-resolved images will then be submitted by the participants and evaluated by the organizers with a set of objective quantitative metrics.


Organizers
Yingqian Wang (wangyingqian16@nudt.edu.cn)
Zhengyu Liang (zyliang@nudt.edu.cn)
Longguang Wang (wanglongguang15@nudt.edu.cn)
Juncheng Li (jcli@cs.ecnu.edu.cn)
Jungang Yang (yangjungang@nudt.edu.cn)
Radu Timofte (Radu.Timofte@vision.ee.ethz.ch)
Yulan Guo (guoyulan@sysu.edu.cn)
Official Repository
NTIRE 2026: Light Field Image Super-Resolution Challenge

Chasuite
Competitions v1.6
Chahub
Chagrade
About
About
Github
Privacy and Terms
API Docs
CodaBench
Join us on Github for contact & bug reports

Questions about the platform? See our Docs for more information.

1.22
Benchmarks/Competitions
Datasets
Competition Logo
NTIRE 2026 Light Field Image Super-Resolution Challenge - Track 2 Efficiency
Organized by: fyzhang
Current Phase Ends: March 5, 2026 at 7:00 PM EST
Current server time: January 31, 2026 at 1:11 PM EST
Docker image: codalab/codalab-legacy:py39 
NTIRE 2026: Light Field Image Super-Resolution Challenge
Evaluation
The final goal of this challenge is to develop methods to enhance the spatial resolution of Light Field (LF) images. During the development and testing phase, challenge participants will submit the super-resolved results.


Track 1: Classic

This track aims to encourage participants to explore the precision upper bound of LF image SR with the given standard training data. In this track, there are no restrictions on efficiency (e.g., parameter amount or computational cost) for the developed models. However, the use of external training data or pretrained models is strictly prohibited. To measure the performance, we use the standard Peak Signal to Noise Ratio (PSNR) and, complementarily, the Structural Similarity (SSIM) index as they are often employed in the literature. PSNR and SSIM implementations can be found in the BasicLFSR toolbox. We report the submissions over all the processed LF images (with the angular resolution of size 5×5), and rank the submissions according to the average PSNR values on the evaluation dataset only. The SSIM metrics will not affect submission rankings but will be used to review the strengths and weaknesses of suggested methods in the final challenge report.


Track 2: Efficiency

This track aims to highlight an under-investigated issue of efficient inference in LF image SR, and hope the participants to develop LF image SR methods that can achieve high computational efficiency without compromising the SR quality. Note that, in this track, the model size (i.e., number of parameters) is restricted to 1 MB, and the FLOPs computed using the fvcore library is restricted to within 20 G (with an input LF of size 5×5×32×32). And the computational cost of Test-Time Augmentation (TTA) operations would be counted to the final FLOPs. Models that do not meet the efficiency requirement will not be included for the ranking. After the submission of the fact sheet and code, we will also assess the inference time of each model as an additional metric for evaluation in the final challenge report. The rankings are determined by the average PSNR value on the test set only.


Track 3: Large Model

In this track, we aspire to leverage the power of large foundation models to boost the performance of LF image SR. In this track, the participants are allowed to use external training data and pretrained models for model development, and there is no efficiency limitation. The rankings are determined by the average PSNR value on the test set only. Note that, validation set and test set are NOT allowed for training.


Submission
During the development phase, the participants can submit their results on the validation set to get feedback from the Codabench server. The validation set should only be used for evaluation and analysis purposes but NOT for training. At the testing phase, the participants will submit the whole restoration results of the test set. This should match the last submission to the Codabench.


Both of the validation and test sets are divided into Real(-world) and Synth(etic) subsets, and each subset consists of 8 LF scenes saved in .mat format. The terminology of the super-resolved results should be identical to that of the original files in the validation and test sets. Note that, all sub-aperture images of each LF require to be saved in .bmp format and saved in one folder named with the original LF scene name, i.e., SCENE_NAME.mat -> SCENE_NAME/View_0_0.bmp ... View_4_4.bmp.


When submitting the results, create a ZIP archive containing all the generated results. Note that, Real and Synth folders should be in the root of the archive. A valid example dummy submission file can be found here.


After the test phase, the participants will submit the fact sheet and source code to the official submission account (ntire.lfsr@outlook.com) by email. The final submission should be made by the following rules:

The submitted results must be from the same method that generated the last submission to the Codabench. We will check the consistency. Otherwise, the submission is invalid.
Both the testing source code (or executable) and the training code must be submitted. Reproducibility is a necessary condition.
Factsheet describing the method details should be submitted together. The factsheet format is provided with data. Participants must submit a compiled pdf file and the tex source. We encourage participants to provide details and enclose figure image sources together. This helps writing the challenge summary report.
Each participating team in the final testing phase should use the following factsheet template to describe their solution(s).

Email submission format
Please use the following format (an example can be found here) to submit your final results, fact sheet, code, model (with trained parameters). We will run the test code to reproduce the results. Training code doesn't necessarily have to be included. The code and the model is to be posted on the NTIRE 2026 website.


to: ntire.lfsr@outlook.com;
cc: your_team_members
title: [NTIRE 2026 Light Field Image Super-Resolution Challenge] - [Track X] - [TEAM_NAME]
body should include:

the challenge name (including Track id)
team name
team leader's name, affiliation, and email address
team members' names, affiliations, and email addresses
user names on NTIRE 2026 Codabench competitions (if any)
executable/source code attached or download links.
fact sheet attached
download link to the results

It should be noted that, the top ranking participants should publicly release their code (or executables) under a license of their choice, taken among popular OSI-approved licenses (http://opensource.org/licenses) and make their code (or executables) online accessible for not less than one year following the end of the challenge (applies only for top three ranked participants of the competition).

Group number policy
Each group cannot have more than six group members (i.e., 1 to 6 group members is OK), and each paricipant can only join one group. Each group can only submit one algorithm for final ranking.


Chasuite
Competitions v1.6
Chahub
Chagrade
About
About
Github
Privacy and Terms
API Docs
CodaBench
Join us on Github for contact & bug reports

Questions about the platform? See our Docs for more information.

1.22
Benchmarks/Competitions
Datasets
Competition Logo
NTIRE 2026 Light Field Image Super-Resolution Challenge - Track 2 Efficiency
Organized by: fyzhang
Current Phase Ends: March 5, 2026 at 7:00 PM EST
Current server time: January 31, 2026 at 1:11 PM EST
Docker image: codalab/codalab-legacy:py39 
NTIRE 2026: Light Field Image Super-Resolution Challenge
Terms and Conditions
These are the official rules (terms and conditions) that govern how the NTIRE 2026 challenge on light field (LF) image super-resolution (SR) will operate. This challenge will be simply refered to as the "challenge" or the "contest" throghout the remaining part of these rules and may be named as "NTIRE" or "LF-SR" benchmark, challenge, or contest, elsewhere (our webpage, our documentation and publications).

In these rules, "we", "our" and "us" refer to the organizers (Yingqian Wang, Zhengyu Liang, Longguang Wang, Juncheng Li, Jungang Yang, Radu Timofte and Yulan Guo) of this challenge, and "you" refers to an eligible contest participant.


Note that, these official rules can change during the contest until the start of the final phase. If at any point during the contest the registered participant considers that they can not anymore meet the eligibility criteria or do not agree with the changes in the official terms and conditions, it is the responsibility of the participant to send an email to the organizers such that to be removed from all the records. Once the contest is over no change is possible in the status of the registered participants and their entries.


1. Contest description
This is a skill-based contest and chance plays no part in the determination of the winner(s).


The goal of the contest is to super-resolve input LF images in the spatial domain with an upsampling factor of 4 (i.e., 4×SR).


Focus of the contest: The publicly-available EPFL, HCInew, HCIold, INRIA and STFgantry datasets will be used to develop algorithms for the challenge (Tracks 1&2&3). LF images in above datasets have a large diversity of contents. The dataset is divided into training and validation sets (note: the test sets in above datasets are not used). The objective of this challenge is to generate super-resolved LF images with high fidelity (PSNR) to the ground truth. During the test phase, the participants will not have access to the ground truth images from the test data. The ranking of the participants is according to the performance of their methods on the test data. The participants will provide descriptions of their methods, details on (running) time complexity, and platform information. The winners will be determined according to their entries, the reproducibility of the results and uploaded codes or executables, and the above mentioned criteria as judged by the organizers.


2. Eligibility
You are eligible to register and compete in this contest only if you meet all the following requirements:

You are an individual or a team of people willing to contribute to the open tasks, who accepts to follow the rules of this contest.
You are not an NTIRE challenge organizer or an employee of NTIRE challenge organizers.
You are not involved in any part of the administration and execution of this contest.
You are not a first-degree relative, partner, household member of an employee or of an organizer of the NTIRE challenge, or of a person involved in any part of the administration and execution of this contest.
This contest is void wherever it is prohibited by law.
Entries submitted but not qualified to enter the contest, it is considered voluntary and for any entry you submit, NTIRE reserves the right to evaluate it for scientific purposes, however, under no circumstances will such entries qualify for sponsored prizes. If you are an employee, affiliated with or representant of any of the NTIRE challenge sponsors then you are allowed to enter in the contest and get ranked, however, if you will rank among the winners with eligible entries you will receive only a diploma award and none of the sponsored money, products or travel grants.

NOTE: Industry and research labs are allowed to submit entries and to compete in both the validation phase and final test phase. However, in order to get officially ranked on the final test leaderboard and to be eligible for awards the reproducibility of the results is a must and, therefore, the participants need to make available and submit their codes or executables. All the top entries will be checked for reproducibility and marked accordingly.


3. Entry
In order to be eligible for judging, an entry must meet all the following requirements:


Entry contents: The participants are required to submit image results and code (or executables). The top ranking participants should publicly release their code (or executables) under a license of their choice, taken among popular OSI-approved licenses (http://opensource.org/licenses) and make their code (or executables) online accessible for a period of not less than one year following the end of the challenge (applies only for top three ranked participants of the competition). To enter the final ranking the participants will need to fill out a survey (fact sheet) briefly describing their method. All the participants are also invited (not mandatory) to submit a paper for peer-reviewing and publication at the NTIRE Workshop and Challenges. The participants' score must improve the baseline performance provided by the challenge organizers.


Use of the provided data: All data provided by NTIRE are freely available to the participants from the website of the challenge under license terms provided with the data. The data are available only for open research and educational purposes, within the scope of the challenge. NTIRE and the organizers make no warranties regarding the database, including but not limited to warranties of non-infringement or fitness for a particular purpose. The copyright of the images remains in the property of their respective owners. By downloading and making use of the data, you accept full responsibility for using the data. You shall defend and indemnify NTIRE and the organizers, including their employees, Trustees, officers and agents, against any and all claims arising from your use of the data. You agree not to redistribute the data without this notice.

Test data: The organizers will use the test data for the final evaluation and ranking of the entries. The ground truth test data will not be made available to the participants during the contest.
<li><b>Training and validation data: </b>The organizers will make available to the participants a training and a validation dataset with ground truth images.<br></li>

<li><b>Post-challenge analyses: </b>The organizers may also perform additional post-challenge analyses using extra-data, but without effect on the challenge ranking.<br></li>

<li><b>Submission: </b>The entries will be online submitted via the Codabench web platform. During the development phase, while the validation server is online, the participants will receive immediate feedback on validation data. The final evaluation will be computed automatically on the test data submissions, but the final scores will be released after the challenge is over.<br></li>

<li><b>Original work, permissions: </b>In addition, by submitting your entry into this contest you confirm that, to the best of your knowledge: - your entry is your own original work; and - your entry only includes material that you own, or that you have permission to use.<br><br></li>
4. Potential use of entry
Other than what is set forth below, we are not claiming any ownership rights to your entry. However, by submitting your entry, you:


Are granting us an irrevocable, worldwide right and license, in exchange for your opportunity to participate in the contest and potential prize awards, for the duration of the protection of the copyrights to:

<li>Use, review, assess, test and otherwise analyze results submitted or produced by your code or executable and other material submitted by you in connection with this contest and any future research or contests by the organizers; and<br></li>
<li>Feature your entry and all its content in connection with the promotion of this contest in all media (now known or later developed);<br></li>
Agree to sign any necessary documentation that may be required for us and our designees to make use of the rights you granted above;


Understand and acknowledge that we and other entrants may have developed or commissioned materials similar or identical to your submission and you waive any claims you may have resulting from any similarities to your entry;


Understand that we cannot control the incoming information you will disclose to our representatives or our co-sponsor’s representatives in the course of entering, or what our representatives will remember about your entry. You also understand that we will not restrict work assignments of representatives or our co-sponsor’s representatives who have had access to your entry. By entering this contest, you agree that use of information in our representatives’ or our co-sponsor’s representatives unaided memories in the development or deployment of our products or services does not create liability for us under this agreement or copyright or trade secret law;


Understand that you will not receive any compensation or credit for use of your entry, other than what is described in these official rules.


If you do not want to grant us these rights to your entry, please do not enter this contest.


5. Submission of entries
The participants will follow the instructions on the Codabench website to submit entries


The participants will be registered as mutually exclusive teams. Each Codabench account is allowed to submit only one single final entry.


The participants must follow the instructions and the rules. We will automatically disqualify incomplete or invalid entries.


6. Judging the entries
The board of NTIRE will select a panel of judges to judge the entries; all judges will be forbidden to enter the contest and will be experts in causality, statistics, machine learning, computer vision, or a related field, or experts in challenge organization. A list of the judges will be made available upon request. The judges will review all eligible entries received and select three winners for each of the two competition tracks based upon the prediction score on test data. The judges will verify that the winners complied with the rules, including that they documented their method by filling out a fact sheet.


The decisions of these judges are final and binding. The distribution of prizes according to the decisions made by the judges will be made within three months after completion of the last round of the contest. If we do not receive a sufficient number of entries meeting the entry requirements, we may, at our discretion based on the above criteria, not award any or all of the contest prizes below. In the event of a tie between any eligible entries, the tie will be broken by giving preference to the earliest submission, using the time stamp of the submission platform.


7. Prizes and awards
The financial sponsors of this contest are listed on NTIRE 2026 workshop web page. There will be economic incentive prizes and travel grants for the winners (based on availability) to boost contest participation; these prizes will not require participants to enter into an IP agreement with any of the sponsors, to disclose algorithms, or to deliver source code to them. The participants affiliated with the industry sponsors agree to not receive any sponsored money, product or travel grant in the case they will be among the winners.


8. Other sponsored events
Publishing papers is optional and will not be a condition to entering the challenge or winning prizes. The top ranking participants are invited to submit a maximum 8-pages paper (CVPR 2026 author rules) for peer-reviewing to NTIRE workshop.


The results of the challenge will be published together with NTIRE 2026 workshop papers in the 2026 CVPR Workshops proceedings.


The top ranked participants and participants contributing interesting and novel methods to the challenge will be invited to be co-authors of the challenge report paper which will be published in the 2026 CVPR Workshops proceedings. A detailed description of the ranked solution as well as the reproducibility of the results are a must to be an eligible co-author.


9. Notifications
If there is any change to data, schedule, instructions of participation, or these rules, the registered participants will be notified at the email they provided with the registration.


Within seven days following the determination of winners, we will send a notification to the potential winners. If the notification that we send is returned as undeliverable, or you are otherwise unreachable for any reason, we may award the prize to an alternate winner, unless forbidden by applicable law.


The prize such as money, product, or travel grant will be delivered to the registered team leader given that the team is not affiliated with any of the sponsors. It's up to the team to share the prize. If this person becomes unavailable for any reason, the prize will be delivered to be the authorized account holder of the e-mail address used to make the winning entry.


If you are a potential winner, we may require you to sign a declaration of eligibility, use, indemnity and liability/publicity release and applicable tax forms. If you are a potential winner and are a minor in your place of residence, and we require that your parent or legal guardian will be designated as the winner, and we may require that they sign a declaration of eligibility, use, indemnity and liability/publicity release on your behalf. If you, (or your parent/legal guardian if applicable), do not sign and return these required forms within the time period listed on the winner notification message, we may disqualify you (or the designated parent/legal guardian) and select an alternate selected winner.


Chasuite
Competitions v1.6
Chahub
Chagrade
About
About
Github
Privacy and Terms
API Docs
CodaBench
Join us on Github for contact & bug reports

Questions about the platform? See our Docs for more information.

1.22
Benchmarks/Competitions
Datasets
Competition Logo
NTIRE 2026 Light Field Image Super-Resolution Challenge - Track 2 Efficiency
Organized by: fyzhang
Current Phase Ends: March 5, 2026 at 7:00 PM EST
Current server time: January 31, 2026 at 1:11 PM EST
Docker image: codalab/codalab-legacy:py39 
NTIRE 2026: Light Field Image Super-Resolution Challenge
Terms and Conditions
These are the official rules (terms and conditions) that govern how the NTIRE 2026 challenge on light field (LF) image super-resolution (SR) will operate. This challenge will be simply refered to as the "challenge" or the "contest" throghout the remaining part of these rules and may be named as "NTIRE" or "LF-SR" benchmark, challenge, or contest, elsewhere (our webpage, our documentation and publications).

In these rules, "we", "our" and "us" refer to the organizers (Yingqian Wang, Zhengyu Liang, Longguang Wang, Juncheng Li, Jungang Yang, Radu Timofte and Yulan Guo) of this challenge, and "you" refers to an eligible contest participant.


Note that, these official rules can change during the contest until the start of the final phase. If at any point during the contest the registered participant considers that they can not anymore meet the eligibility criteria or do not agree with the changes in the official terms and conditions, it is the responsibility of the participant to send an email to the organizers such that to be removed from all the records. Once the contest is over no change is possible in the status of the registered participants and their entries.


1. Contest description
This is a skill-based contest and chance plays no part in the determination of the winner(s).


The goal of the contest is to super-resolve input LF images in the spatial domain with an upsampling factor of 4 (i.e., 4×SR).


Focus of the contest: The publicly-available EPFL, HCInew, HCIold, INRIA and STFgantry datasets will be used to develop algorithms for the challenge (Tracks 1&2&3). LF images in above datasets have a large diversity of contents. The dataset is divided into training and validation sets (note: the test sets in above datasets are not used). The objective of this challenge is to generate super-resolved LF images with high fidelity (PSNR) to the ground truth. During the test phase, the participants will not have access to the ground truth images from the test data. The ranking of the participants is according to the performance of their methods on the test data. The participants will provide descriptions of their methods, details on (running) time complexity, and platform information. The winners will be determined according to their entries, the reproducibility of the results and uploaded codes or executables, and the above mentioned criteria as judged by the organizers.


2. Eligibility
You are eligible to register and compete in this contest only if you meet all the following requirements:

You are an individual or a team of people willing to contribute to the open tasks, who accepts to follow the rules of this contest.
You are not an NTIRE challenge organizer or an employee of NTIRE challenge organizers.
You are not involved in any part of the administration and execution of this contest.
You are not a first-degree relative, partner, household member of an employee or of an organizer of the NTIRE challenge, or of a person involved in any part of the administration and execution of this contest.
This contest is void wherever it is prohibited by law.
Entries submitted but not qualified to enter the contest, it is considered voluntary and for any entry you submit, NTIRE reserves the right to evaluate it for scientific purposes, however, under no circumstances will such entries qualify for sponsored prizes. If you are an employee, affiliated with or representant of any of the NTIRE challenge sponsors then you are allowed to enter in the contest and get ranked, however, if you will rank among the winners with eligible entries you will receive only a diploma award and none of the sponsored money, products or travel grants.

NOTE: Industry and research labs are allowed to submit entries and to compete in both the validation phase and final test phase. However, in order to get officially ranked on the final test leaderboard and to be eligible for awards the reproducibility of the results is a must and, therefore, the participants need to make available and submit their codes or executables. All the top entries will be checked for reproducibility and marked accordingly.


3. Entry
In order to be eligible for judging, an entry must meet all the following requirements:


Entry contents: The participants are required to submit image results and code (or executables). The top ranking participants should publicly release their code (or executables) under a license of their choice, taken among popular OSI-approved licenses (http://opensource.org/licenses) and make their code (or executables) online accessible for a period of not less than one year following the end of the challenge (applies only for top three ranked participants of the competition). To enter the final ranking the participants will need to fill out a survey (fact sheet) briefly describing their method. All the participants are also invited (not mandatory) to submit a paper for peer-reviewing and publication at the NTIRE Workshop and Challenges. The participants' score must improve the baseline performance provided by the challenge organizers.


Use of the provided data: All data provided by NTIRE are freely available to the participants from the website of the challenge under license terms provided with the data. The data are available only for open research and educational purposes, within the scope of the challenge. NTIRE and the organizers make no warranties regarding the database, including but not limited to warranties of non-infringement or fitness for a particular purpose. The copyright of the images remains in the property of their respective owners. By downloading and making use of the data, you accept full responsibility for using the data. You shall defend and indemnify NTIRE and the organizers, including their employees, Trustees, officers and agents, against any and all claims arising from your use of the data. You agree not to redistribute the data without this notice.

Test data: The organizers will use the test data for the final evaluation and ranking of the entries. The ground truth test data will not be made available to the participants during the contest.
<li><b>Training and validation data: </b>The organizers will make available to the participants a training and a validation dataset with ground truth images.<br></li>

<li><b>Post-challenge analyses: </b>The organizers may also perform additional post-challenge analyses using extra-data, but without effect on the challenge ranking.<br></li>

<li><b>Submission: </b>The entries will be online submitted via the Codabench web platform. During the development phase, while the validation server is online, the participants will receive immediate feedback on validation data. The final evaluation will be computed automatically on the test data submissions, but the final scores will be released after the challenge is over.<br></li>

<li><b>Original work, permissions: </b>In addition, by submitting your entry into this contest you confirm that, to the best of your knowledge: - your entry is your own original work; and - your entry only includes material that you own, or that you have permission to use.<br><br></li>
4. Potential use of entry
Other than what is set forth below, we are not claiming any ownership rights to your entry. However, by submitting your entry, you:


Are granting us an irrevocable, worldwide right and license, in exchange for your opportunity to participate in the contest and potential prize awards, for the duration of the protection of the copyrights to:

<li>Use, review, assess, test and otherwise analyze results submitted or produced by your code or executable and other material submitted by you in connection with this contest and any future research or contests by the organizers; and<br></li>
<li>Feature your entry and all its content in connection with the promotion of this contest in all media (now known or later developed);<br></li>
Agree to sign any necessary documentation that may be required for us and our designees to make use of the rights you granted above;


Understand and acknowledge that we and other entrants may have developed or commissioned materials similar or identical to your submission and you waive any claims you may have resulting from any similarities to your entry;


Understand that we cannot control the incoming information you will disclose to our representatives or our co-sponsor’s representatives in the course of entering, or what our representatives will remember about your entry. You also understand that we will not restrict work assignments of representatives or our co-sponsor’s representatives who have had access to your entry. By entering this contest, you agree that use of information in our representatives’ or our co-sponsor’s representatives unaided memories in the development or deployment of our products or services does not create liability for us under this agreement or copyright or trade secret law;


Understand that you will not receive any compensation or credit for use of your entry, other than what is described in these official rules.


If you do not want to grant us these rights to your entry, please do not enter this contest.


5. Submission of entries
The participants will follow the instructions on the Codabench website to submit entries


The participants will be registered as mutually exclusive teams. Each Codabench account is allowed to submit only one single final entry.


The participants must follow the instructions and the rules. We will automatically disqualify incomplete or invalid entries.


6. Judging the entries
The board of NTIRE will select a panel of judges to judge the entries; all judges will be forbidden to enter the contest and will be experts in causality, statistics, machine learning, computer vision, or a related field, or experts in challenge organization. A list of the judges will be made available upon request. The judges will review all eligible entries received and select three winners for each of the two competition tracks based upon the prediction score on test data. The judges will verify that the winners complied with the rules, including that they documented their method by filling out a fact sheet.


The decisions of these judges are final and binding. The distribution of prizes according to the decisions made by the judges will be made within three months after completion of the last round of the contest. If we do not receive a sufficient number of entries meeting the entry requirements, we may, at our discretion based on the above criteria, not award any or all of the contest prizes below. In the event of a tie between any eligible entries, the tie will be broken by giving preference to the earliest submission, using the time stamp of the submission platform.


7. Prizes and awards
The financial sponsors of this contest are listed on NTIRE 2026 workshop web page. There will be economic incentive prizes and travel grants for the winners (based on availability) to boost contest participation; these prizes will not require participants to enter into an IP agreement with any of the sponsors, to disclose algorithms, or to deliver source code to them. The participants affiliated with the industry sponsors agree to not receive any sponsored money, product or travel grant in the case they will be among the winners.


8. Other sponsored events
Publishing papers is optional and will not be a condition to entering the challenge or winning prizes. The top ranking participants are invited to submit a maximum 8-pages paper (CVPR 2026 author rules) for peer-reviewing to NTIRE workshop.


The results of the challenge will be published together with NTIRE 2026 workshop papers in the 2026 CVPR Workshops proceedings.


The top ranked participants and participants contributing interesting and novel methods to the challenge will be invited to be co-authors of the challenge report paper which will be published in the 2026 CVPR Workshops proceedings. A detailed description of the ranked solution as well as the reproducibility of the results are a must to be an eligible co-author.


9. Notifications
If there is any change to data, schedule, instructions of participation, or these rules, the registered participants will be notified at the email they provided with the registration.


Within seven days following the determination of winners, we will send a notification to the potential winners. If the notification that we send is returned as undeliverable, or you are otherwise unreachable for any reason, we may award the prize to an alternate winner, unless forbidden by applicable law.


The prize such as money, product, or travel grant will be delivered to the registered team leader given that the team is not affiliated with any of the sponsors. It's up to the team to share the prize. If this person becomes unavailable for any reason, the prize will be delivered to be the authorized account holder of the e-mail address used to make the winning entry.


If you are a potential winner, we may require you to sign a declaration of eligibility, use, indemnity and liability/publicity release and applicable tax forms. If you are a potential winner and are a minor in your place of residence, and we require that your parent or legal guardian will be designated as the winner, and we may require that they sign a declaration of eligibility, use, indemnity and liability/publicity release on your behalf. If you, (or your parent/legal guardian if applicable), do not sign and return these required forms within the time period listed on the winner notification message, we may disqualify you (or the designated parent/legal guardian) and select an alternate selected winner.


Chasuite
Competitions v1.6
Chahub
Chagrade
About
About
Github
Privacy and Terms
API Docs
CodaBench
Join us on Github for contact & bug reports

Questions about the platform? See our Docs for more information.

1.22