import subprocess
import os
import tempfile
import uuid
import shutil

class DockerRunner:
    def __init__(self, image_name="hire-evaluator"):
        self.image_name = image_name

    def run(self, files, command, timeout=15):
        """
        Runs a command in a Docker container with the specified files.
        :param files: A dictionary mapping filenames to their content.
        :param command: The command to run in the container.
        :param timeout: Execution timeout in seconds.
        :return: (stdout, stderr, returncode)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files in the temporary directory
            for filename, content in files.items():
                file_path = os.path.join(tmpdir, filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)

            # Construct the docker command
            # Using --rm to remove the container after exit
            # Using -v to mount the temp directory to /app
            # Using -w to set the working directory to /app
            docker_cmd = [
                "docker", "run", "--rm",
                "-v", f"{os.path.abspath(tmpdir)}:/app",
                "-w", "/app",
                self.image_name,
                "sh", "-c", command
            ]

            try:
                result = subprocess.run(
                    docker_cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )
                return result.stdout, result.stderr, result.returncode
            except subprocess.TimeoutExpired as e:
                # Cleanup if timed out (though --rm should handle it, we want to return a specific state)
                return "", "Execution timed out", -1
            except Exception as e:
                return "", str(e), -2
