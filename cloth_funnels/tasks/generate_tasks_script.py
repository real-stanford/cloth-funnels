
if __name__ == "__main__":
    import sys
    category = sys.argv[1]
    train_path = f'assets/tasks/multi-{category}-train.hdf5'
    eval_path = f'assets/tasks/multi-{category}-eval.hdf5'
    num_processes = 8

    prefix = "CUDA_VISIBLE_DEVICES=0,5,6,7"

    task_difficulties = [['hard', 1500], ['easy', 500]]
    eval_difficulties = [['hard', 200], ['easy', 200]]

    task_difficulty_settings = {
        'hard':'--randomize_direction --random_translation 0.3',
        'easy':' --random_translation 0.05'
    }


    for i, tup in enumerate(task_difficulties):
        if i == 0: continue
        task_difficulties[i][1] += task_difficulties[i-1][1]
        eval_difficulties[i][1] += eval_difficulties[i-1][1]

    final_str = ""
    for i, tup in enumerate(task_difficulties):
        diff_str = f"{prefix} python cloth_funnels/tasks/generate_tasks.py --mesh_category {category}.json --path {train_path} " +\
            f"--task unfold --num_processes {num_processes} --task_difficulty {task_difficulties[i][0]} "+\
            f" --num_tasks {task_difficulties[i][1]} {task_difficulty_settings[task_difficulties[i][0]]}"
        final_str += diff_str + " &&"

    for i, tup in enumerate(eval_difficulties):
        diff_str = f"{prefix} python cloth_funnels/tasks/generate_tasks.py --mesh_category {category}.json --path {eval_path} " +\
            f"--task unfold --num_processes {num_processes} --eval --task_difficulty {eval_difficulties[i][0]} "+\
                f"{task_difficulty_settings[task_difficulties[i][0]]}"+\
            f" --num_tasks {eval_difficulties[i][1]}"
        final_str += diff_str
        if i != len(eval_difficulties) - 1:
            final_str += " &&"
            
    print(final_str)
