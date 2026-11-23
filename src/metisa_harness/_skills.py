"""Discover the available skills."""

from __future__ import annotations

from pathlib import Path

import yaml

from .models import AgentDefinition, AgentSkill


def get_skill_catalog(
    agent: AgentDefinition,
) -> list[AgentSkill]:
    """Get the skill catalog for the specified agent."""
    skills: list[AgentSkill] = []
    if agent.skills_directory is None:
        return skills

    if not agent.skills_directory.is_dir():
        raise ValueError("Invalid agent skills directory.")

    for child in sorted(agent.skills_directory.iterdir()):
        if child.is_dir():
            skill = get_skill(child)
            if skill:
                skills.append(skill)

    return skills


def get_skill(
    directory: Path,
) -> AgentSkill | None:
    """Get the skill in the specified directory."""
    if not directory.is_dir():
        return None

    skill_md = directory / "SKILL.md"
    if not skill_md.is_file():
        return None

    skills_definition = skill_md.read_text(encoding="utf-8")

    if not skills_definition.startswith("---"):
        raise RuntimeError("Invalid skill.md file.")

    # Split into three parts:
    # - empty string before first ---
    # - YAML front matter
    # - Markdown content
    parts = skills_definition.split("---")
    if len(parts) != 3:
        raise RuntimeError("Invalid skill.md file.")

    frontmatter = yaml.safe_load(parts[1]) or {}
    content = parts[2].strip()

    name = frontmatter.get("name")
    if name is None or not name.strip():
        raise RuntimeError("Invalid skill.md file: missing name.")
    description = frontmatter.get("description")
    if description is None or not description.strip():
        raise RuntimeError("Invalid skill.md file: missing description.")

    if not content:
        raise RuntimeError("Invalid skill.md file: missing content.")

    return AgentSkill(
        name=name,
        description=description,
        content=content,
        directory=directory,
    )
