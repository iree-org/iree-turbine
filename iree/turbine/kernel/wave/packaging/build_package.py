import pathlib
from typing import Any
import jinja2
import shutil


def build_folders(kernel_info: dict[str, Any], output_dir: str):
    package_path = pathlib.Path(output_dir) / kernel_info["package_name"]
    package_path.mkdir(parents=True, exist_ok=True)
    subfolder = package_path / kernel_info["package_name"]
    subfolder.mkdir(parents=True, exist_ok=True)
    init_file = subfolder / "__init__.py"
    with open(init_file, "w") as f:
        f.write(f"from .main import {kernel_info['kernel_name']}\n")
    return subfolder


def copy_artifacts(kernel_info: dict[str, Any], output_dir: str):
    shutil.copy(kernel_info["vmfb_path"], output_dir)


def render_templates(kernel_info: dict[str, Any], output_dir: str):
    parent_dir = pathlib.Path(__file__).resolve().parent
    template_loader = jinja2.FileSystemLoader(searchpath=parent_dir / "templates")
    template_env = jinja2.Environment(loader=template_loader)
    main_template = template_env.get_template("main.py.j2")
    updated_template = main_template.render(
        kernel_function_name=kernel_info["kernel_name"],
        kernel_num_inputs=kernel_info["num_inputs"],
        kernel_dispatch_name=kernel_info["dispatch_name"],
        vmfb_path=pathlib.Path(kernel_info["vmfb_path"]).name,
    )
    with open(output_dir / "main.py", "w") as f:
        f.write(updated_template)
    setup_template = template_env.get_template("setup.py.j2")
    updated_template = setup_template.render(
        kernel_package_name=kernel_info["package_name"],
        kernel_version=kernel_info["kernel_version"],
    )
    with open(output_dir.parents[0] / "setup.py", "w") as f:
        f.write(updated_template)


def create_pip_package(kernel_info: dict[str, Any], output_dir: str):
    """Builds a pip package from the current directory."""

    subfolder = build_folders(kernel_info, output_dir)
    copy_artifacts(kernel_info, subfolder)
    render_templates(kernel_info, subfolder)
