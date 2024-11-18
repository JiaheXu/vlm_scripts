import os
from tqdm import tqdm
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import time
import google.generativeai as genai
import json
import pdb
import cv2
import numpy as np


def generate_prompt(task_description, scene_description, traj_features_str):
    return f"""
            Annotate the training trajectory with reasoning

            ## Specification of the experimental setup
            You’re an expert reinforcement learning researcher. You’ve trained an optimal policy for controlling a robotic arm. The
            robot successfully completed a task specified by the instruction: "{task_description}". You are watching a playback of that
            successful task completion.  This video shows two camera views of the task completion, on the left side we look over the shoulder of the robot
            and on the right side is a video from the robot's wrist.  In this playback, the robotic arm executes a sequence of actions. The sequences between important keyframes 
            in the robot's trajectory can be characterized by the following features:

            ‘‘‘ python
                trajectory_features = {traj_features_str}
            ‘‘‘

            Each key in that dictionary corresponds to a frame sequence in the video "[start frame, end frame)".  The values
            describes the low level moves the robot makes between those keyframes.

            ## Scene description

            The robot is operating in the following environment. {scene_description}

            ## Your objective

            I want you to annotate the given trajectory with reasoning. That is, for each frame sequence in trajectory_features, I need to know the action
            that is taken, the relevent objects, and the reasoning that justifies the action choice.  
            
            ### Begin by describing the task [TASK DESCRIPTION]

            Start by giving an overview of the task. Think about the instruction, the trajectory features, as well as the environment.  Make it more comprehensive than the 
            simple instruction. Include the activity,the objects the robotic arm interacts with, and their relative locations in the environment.  This should be a numbered list of
            steps that the robot must take to complete the task.Place this description inside a tag <task>.

            ### Describe the reasoning and actions for each frame sequence

            #### Reasoning [REASON]

            For each frame sequence, you should reason about what the robot is doing.  You should consider the objects in the current frames and the task description.
            You should think about what has happened in the previous frames and what needs to happen in the future.  [REASONING] should be a single string in natural langue
            that describes the reasoning.  It should include all the relevant features from the task description and the scene description.  Place this reasoning inside a tag <reasoning>.
            <reasoning>[REASONING]</reasoning>

            #### Action [ACTION]
            Based on the reasoning, you should succinctly describe the action that is being taken.  This should include the action that is being taken (i.e. move, grasp, rotate, pull, etc.).  
            If relevent, explicitly state the object that is being acted upon (i.e. pull drawer, grasp obj1, release obj2, etc.).  If the robot is moving, explicitly state the target 
            of that motion (e.g. to the left side of the table, above the basket, near the door, etc.).  If the robot is not doing anything, you should state the action as "stop".  
            Place the action choice inside a tag <action>[ACTION]</action>.

            ## Task summary

            Here is a breakdown of what needs to be done:

            - First you must describe the task in detail.
            -Then state the high-level actions that were executed, based on the completed task and the listed features.  These should describe what happens between each frame sequence.
            - For each stated aciton, describe the reasoning that leads to determining the correct action. The reasoning should
            be descriptive and precise. You should provide exactly one reasoning string for each step on the trajectory specified by
            ‘trajectory_features‘.
            - At the very end of the response, write a single label FINISHED to indicate that the answer is complete.

            ## Skeleton

            Here is a skeleton of what the output should look like:

            <task>[TASK DESCRIPTION]</task>
            {{
                (frame_a, frame_b):<reasoning>[REASONING]</reasoning><action>[ACTION]</action>,
                (frame_b, frame_c):<reasoning>[REASONING]</reasoning><action>[ACTION]</action>,
                (frame_c, frame_d):<reasoning>[REASONING]</reasoning><action>[ACTION]</action>,
                ...
            }}
        """    

def classify_movement(trajectory, threshold=0.03, step_ahead=14):
    """
    Classify the robot's movement based on the difference between the current state and the state
    four steps ahead, using the provided threshold for each axis.
    
    Args:
        trajectory (list of tuples): A list where each entry is a tuple (x, y, z, tilt, rotation, gripper_state).
        threshold (float): The movement threshold for each axis.
        step_ahead (int): Number of steps to look ahead for comparison.
    
    Returns:
        dict: A dictionary with step numbers as keys and the corresponding movement labels as values.
    """

    labels = {}
    
    num_steps = len(trajectory)
    
    count = 0
    for i in range(0, num_steps, step_ahead):
        if i + step_ahead < num_steps:
            current_state = np.array(trajectory[i])
            future_state = np.array(trajectory[i + step_ahead])
            diff = future_state - current_state
            
            # Initialize movement description
            movement = []
            
            # Check the movement along x (forward/backward), y (left/right), z (up/down)
            if diff[0] > threshold:
                # movement.append("forward")
                movement.append("move")
            elif diff[0] < -threshold:
                # movement.append("backward")
                movement.append("move")
            if diff[1] < threshold:
                # movement.append("right")
                movement.append("move")
            elif diff[1] > -threshold:
                # movement.append("left")
                movement.append("move")
            if diff[2] > threshold:
                # movement.append("up")
                movement.append("move")
            elif diff[2] < -threshold:
                # movement.append("down")
                movement.append("move")
            
            # Check tilt (assumed to be in the 4th position in the tuple)
            if diff[3] > threshold*8:
                # movement.append("tilt_up")
                movement.append("rotate")
            elif diff[3] < -threshold*8:
                # movement.append("tilt_down")
                movement.append("rotate")
            
            # Check rotation (assumed to be in the 5th position in the tuple)
            if diff[4] > threshold*8:
                # movement.append("rotate_clockwise")
                movement.append("rotate")
            elif diff[4] < -threshold*8:
                # movement.append("rotate_counterclockwise")
                movement.append("rotate")
            
            # Check gripper (assumed to be in the 7th position in the tuple)
            if diff[6] > threshold:
                movement.append("open_gripper")
            elif diff[6] < -threshold:
                movement.append("close_gripper")
            
            # If no movement was detected, label it as "stop"
            if not movement:
                labels[count] = "stop"
            else:
                # labels[count] = "move " + " ".join(movement)
                # remove any duplicate words from movement (keep first occurence)
                labels[count] = " ".join(dict.fromkeys(movement))
        else:
            # For the last few steps where we can't look ahead, default to "stop"
            labels[count] = "stop"

        count += 1
    
    return labels

def format_for_prompt_output(trajectory_features):
    """
    Formats the trajectory_features dictionary for input into the earlier prompt.
    
    Args:
        trajectory_features (dict): Dictionary containing movement labels per step.
    
    Returns:
        str: Formatted string representation for the prompt.
    """
    formatted_output = "{\n"
    for step, label in trajectory_features.items():
        formatted_output += f"    {step}: \"{label} \",\n"
    formatted_output += "}"
    return formatted_output


def gemini_CoT(all_video_files,
            output_json_names = None,
            ):
    
    print(f'[INFO] {len(all_video_files)} video files found')

    # TODO(developer): Update the GOOGLE_API_KEY and the project_id 
    GOOGLE_API_KEY='AIzaSyAMcwWOQ17KKXvgnBPpXb92VwCZlkcphfk'
    genai.configure(api_key=GOOGLE_API_KEY)
    project_id = "geminiplanning"
    vertexai.init(project=project_id, location="us-central1")

    # Set up model
    # Set the model to Gemini 1.5 Pro.
    # model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")
    # model = GenerativeModel("gemini-1.5-flash-002")
    model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

    output_json_datas = [{}, {}, {}]
    for i, output_json_name in enumerate(output_json_names):
        if os.path.exists(output_json_name):
            with open(output_json_name, 'r') as f:
                output_json_datas[i] = json.load(f)
            print(f'[INFO] {len(output_json_datas[i] )} video files have been processed')
        
    for i, file_name in enumerate(all_video_files):
        chat_model = model.start_chat()
        has_processed = [(file_name in output_json_data) for output_json_data in output_json_datas]
        has_processed = all(has_processed)
        if has_processed:
            print(f'[INFO] {file_name} has been processed')
            continue

        task_description = file_name.split("/")[-1].replace(".mp4", "").replace("_", " ")

        # get first frame of video
        cap = cv2.VideoCapture(file_name)
        ret, frame = cap.read()

        left_frame = frame[:, :frame.shape[1]//2]
        cap.release()
        # save first frame as image
        cv2.imwrite("first_frame.jpg", left_frame)
        # get uri for first frame
        # upload first frame to gemini
        
        image = genai.upload_file(path="first_frame.jpg")
        # image_file = Part.from_uri(
        #     image.uri, "image/jpeg"
        # )
        scene_prompt = f"""The robot task is: {task_description}.  Briefly describe the things in this scene.  
        Start by explicitly numbering each item that you see.  In parentheses, note the state of the object if obvious and task relevent (open, closed, full, broken, etc.).  
        After this, mention the spatial relationships between the objects.  Finally, state all the objects that are task relevant."""
        # get caption for first frame
        # scene_description = model.generate_content([image_file, scene_prompt])
        scene_description = chat_model.send_message([scene_prompt, image], request_options={"timeout": 600})

        # load npy file with same name as video file
        trajectory = np.load(file_name.replace(".mp4", ".npy"))
        trajectory_features = classify_movement(trajectory, step_ahead=20)
        trajectory_features_str = format_for_prompt_output(trajectory_features)


        keyframes_trajectory_features_str = format_for_prompt_output(trajectory_features_str)

        # # keyframes same name as video file but _keyframes.npy instead of .mp4
        keyframes = np.load(file_name.replace(".mp4", "_keyframes.npy"))
        keyframes = [int(round(k/20.0)) for k in keyframes]
        keyframes.append(len(trajectory_features))
        keyframes = list(set(keyframes))

        # # sort keyframes in ascending order
        keyframes.sort()

        keyframes_trajectory_features = {}
        # # key should be tuple of keyframes (start, end) value should be concatenated string of actions from all trajectory_features that fall within that range
        for i, key in enumerate(keyframes[:-1]):
            next_key = keyframes[i+1]
            trajectory_features_str = ""
            for dex in range(key, next_key):
                if dex in trajectory_features:
                    trajectory_features_str += trajectory_features[dex] + " "
            
            # remove any duplicate words from trajectory_features_str (keep first occurence)
            trajectory_features_str = " ".join(dict.fromkeys(trajectory_features_str.split()))

            # if more than 1 word in trajectory_features_str, remove any instances of "stop"
            if len(trajectory_features_str.split()) > 1:
                trajectory_features_str = trajectory_features_str.replace("stop", "")

            keyframes_trajectory_features[(key, next_key)] = trajectory_features_str

        # # convert keyframes_trajectory_features to string
        keyframes_trajectory_features_str = format_for_prompt_output(keyframes_trajectory_features)

        prompt = generate_prompt(task_description, scene_description.candidates[0].content.parts[0].text, keyframes_trajectory_features_str)
        # prompt = generate_prompt(task_description, scene_description, trajectory_features_str)
        print(prompt)
        # get video_uri
        video_file = genai.upload_file(path=file_name)
        while video_file.state.name == "PROCESSING":
            print('.', end='')
            time.sleep(10)
            video_file = genai.get_file(video_file.name)
        if video_file.state.name == "FAILED":
            raise ValueError(video_file.state.name)
        print(f"Retrieved file '{video_file.display_name}' as: {video_file.uri}")
        # response = model.generate_content([Part.from_uri(video_file.uri, "video/mp4"), prompt])
        response = chat_model.send_message([prompt, video_file], request_options={"timeout": 600})

        # Alternative solution
        text_output_ = response.candidates[0].content.parts[0].text
        output_json_datas[0][file_name] = text_output_
        with open(output_json_names[0], 'w') as f:
            json.dump(output_json_datas[0], f, indent=4)

        time.sleep(1)
        genai.delete_file(video_file.name)
        genai.delete_file(image.name)

def test_gemini(root_dir = "/Users/mbronars/workspace/CMU/Abstractions",
                depth_root_dir = "/Users/mbronars/workspace/CMU/Abstractions",
                video_caption_file = None,
                with_depth_video = False,
                with_postion_video = False,
                with_additional_captions = [0],
                output_json_names = None,
                prompts = None   ):
    print("Start")
    all_videos_dir = os.listdir(root_dir)
    all_videos_dir.sort()
    all_video_files = []
    all_depth_json_files = []
    all_depth_video_files = []
    all_captions = []
    if video_caption_file is None:
        for i in all_videos_dir:
            file_name = os.path.join(root_dir, i, "output.mp4")
            all_video_files.append(file_name)
            file_name = os.path.join(depth_root_dir, i, "object_positions.json")
            all_depth_json_files.append(file_name)
            depth_file_name = os.path.join(depth_root_dir, i, f"{i}.mp4")
            all_depth_video_files.append(depth_file_name)
    else:
        with open(video_caption_file) as f:
             video_files_data = json.load(f)
        for k,v in video_files_data.items():
            all_video_files.append(k)
            all_captions.append(v)
    gemini_CoT(all_video_files,
               output_json_names = output_json_names,
               )

# 2 Round
if __name__ == "__main__":
    save_root = "/Users/mbronars/workspace/CMU/Abstractions/save"
    os.makedirs(save_root, exist_ok=True)
    output_json_names = [
                        f'{save_root}/3DVLA_cutting_board.json',
                        ]
    
    
    test_gemini(root_dir = "/Users/mbronars/workspace/CMU/Abstractions",
                video_caption_file = "/Users/mbronars/workspace/CMU/Abstractions/data/jsons/RLBench_keyframe_caption_large_orig.json",
                with_additional_captions = [0],
                output_json_names = output_json_names,
                prompts = None)
