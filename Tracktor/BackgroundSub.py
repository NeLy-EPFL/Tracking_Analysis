import cv2
import random
from pathlib import Path


DataPath = Path ("/mnt/labserver/DURRIEU_Matthias/Experimental_data/Optogenetics/Optobot/MultiMazeBiS_15_Steel_Wax/Female_Starved_noWater/221116")
Filters = ['Results', 'BadExp', 'Trimmed', 'p6-0_80fps.mp4']

Files = [path for path in DataPath.rglob('*.mp4') if any(match in path.as_posix() for match in Filters) is False]

for file in Files:
        print (file.as_posix())

        input_vidpath = file.as_posix()
        output_vidpath = str(file).replace(file.stem, file.stem + "_tracked")
        codec = "mp4v"

        fgbg5 = cv2.bgsegm.createBackgroundSubtractorGSOC(
        )

        cap = cv2.VideoCapture(input_vidpath)

        scaling = 1.0

        # Set framesize as the same one as the images read from input video
        BG_framesize = (
            int(cap.read()[1].shape[1] * scaling),
            int(cap.read()[1].shape[0] * scaling),
        )

        fourcc = cv2.VideoWriter_fourcc(*codec)
        # Create a Video writer with the desired parameters
        Background_Generator = cv2.VideoWriter(
            filename=file.parent.joinpath("Background_Generator.mp4").as_posix(),
            fourcc=fourcc,
            fps=80,
            frameSize=BG_framesize,
            # isColor=True,
        )

        # Write a video with random frames taken in the input video
        f = 0
        while f <= 300:
            # get total number of frames
            totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            randomFrameNumber = random.randint(0, totalFrames)
            # set frame position
            cap.set(cv2.CAP_PROP_POS_FRAMES, randomFrameNumber)
            success, image = cap.read()

            if success:

                Background_Generator.write(image)

            f += 1

        cap.release()
        Background_Generator.release()
        cv2.destroyAllWindows()

        # Adjust using live rendering to get a clean background image. Default : 500

        cap = cv2.VideoCapture(file.parent.joinpath( "Background_Generator.mp4").as_posix())

        target = 0
        cap.set(
            1, target
        )  # Set the starting point, try to find a section where the fly moves a lot.

        Frame = target + 300

        while 1:
            # read frames
            ret, img = cap.read()
            this = cap.get(1)

            # apply mask for background subtraction
            # fgbg5 is a GSOC background subtraction algorithm
            fgmask5 = fgbg5.apply(img)

            bg = fgbg5.getBackgroundImage()

            #cv2.imshow("Original", img)
            #cv2.imshow("GSOC", fgmask5)
            #cv2.imshow("background", bg)
            subtracted = cv2.absdiff(img, bg)

            if this == Frame:

                cv2.imwrite(file.parent.joinpath(file.stem+"_Background.jpg").as_posix(), bg)
                break

            k = cv2.waitKey(30) & 0xFF
            if k == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        Background = cv2.imread(file.parent.joinpath("Background.jpg").as_posix())
        file.parent.joinpath("Background_Generator.mp4").unlink()