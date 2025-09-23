import asyncio
from datetime import datetime
import json
import random
import re
from python.helpers.tool import Tool, Response
from python.helpers.task_scheduler import (
    TaskScheduler, ScheduledTask, AdHocTask, PlannedTask,
    serialize_task, TaskState, TaskSchedule, TaskPlan, parse_datetime, serialize_datetime
)
from agent import AgentContext
from python.helpers import persist_chat

DEFAULT_WAIT_TIMEOUT = 300


class SchedulerTool(Tool):

    async def execute(self, **kwargs):
        if self.method == "list_tasks":
            return await self.list_tasks(**kwargs)
        elif self.method == "find_task_by_name":
            return await self.find_task_by_name(**kwargs)
        elif self.method == "show_task":
            return await self.show_task(**kwargs)
        elif self.method == "run_task":
            return await self.run_task(**kwargs)
        elif self.method == "delete_task":
            return await self.delete_task(**kwargs)
        elif self.method == "create_scheduled_task":
            return await self.create_scheduled_task(**kwargs)
        elif self.method == "create_adhoc_task":
            return await self.create_adhoc_task(**kwargs)
        elif self.method == "create_planned_task":
            return await self.create_planned_task(**kwargs)
        elif self.method == "wait_for_task":
            return await self.wait_for_task(**kwargs)
        elif self.method == "prioritize_tasks_ecan":
            return await self.prioritize_tasks_ecan(**kwargs)
        else:
            return Response(message=f"Unknown method '{self.name}:{self.method}'", break_loop=False)

    async def list_tasks(self, **kwargs) -> Response:
        state_filter: list[str] | None = kwargs.get("state", None)
        type_filter: list[str] | None = kwargs.get("type", None)
        next_run_within_filter: int | None = kwargs.get("next_run_within", None)
        next_run_after_filter: int | None = kwargs.get("next_run_after", None)

        tasks: list[ScheduledTask | AdHocTask | PlannedTask] = TaskScheduler.get().get_tasks()
        filtered_tasks = []
        for task in tasks:
            if state_filter and task.state not in state_filter:
                continue
            if type_filter and task.type not in type_filter:
                continue
            if next_run_within_filter and task.get_next_run_minutes() is not None and task.get_next_run_minutes() > next_run_within_filter:  # type: ignore
                continue
            if next_run_after_filter and task.get_next_run_minutes() is not None and task.get_next_run_minutes() < next_run_after_filter:  # type: ignore
                continue
            filtered_tasks.append(serialize_task(task))

        return Response(message=json.dumps(filtered_tasks, indent=4), break_loop=False)

    async def find_task_by_name(self, **kwargs) -> Response:
        name: str = kwargs.get("name", None)
        if not name:
            return Response(message="Task name is required", break_loop=False)
        tasks: list[ScheduledTask | AdHocTask | PlannedTask] = TaskScheduler.get().find_task_by_name(name)
        if not tasks:
            return Response(message=f"Task not found: {name}", break_loop=False)
        return Response(message=json.dumps([serialize_task(task) for task in tasks], indent=4), break_loop=False)

    async def show_task(self, **kwargs) -> Response:
        task_uuid: str = kwargs.get("uuid", None)
        if not task_uuid:
            return Response(message="Task UUID is required", break_loop=False)
        task: ScheduledTask | AdHocTask | PlannedTask | None = TaskScheduler.get().get_task_by_uuid(task_uuid)
        if not task:
            return Response(message=f"Task not found: {task_uuid}", break_loop=False)
        return Response(message=json.dumps(serialize_task(task), indent=4), break_loop=False)

    async def run_task(self, **kwargs) -> Response:
        task_uuid: str = kwargs.get("uuid", None)
        if not task_uuid:
            return Response(message="Task UUID is required", break_loop=False)
        task_context: str | None = kwargs.get("context", None)
        task: ScheduledTask | AdHocTask | PlannedTask | None = TaskScheduler.get().get_task_by_uuid(task_uuid)
        if not task:
            return Response(message=f"Task not found: {task_uuid}", break_loop=False)
        await TaskScheduler.get().run_task_by_uuid(task_uuid, task_context)
        if task.context_id == self.agent.context.id:
            break_loop = True  # break loop if task is running in the same context, otherwise it would start two conversations in one window
        else:
            break_loop = False
        return Response(message=f"Task started: {task_uuid}", break_loop=break_loop)

    async def delete_task(self, **kwargs) -> Response:
        task_uuid: str = kwargs.get("uuid", None)
        if not task_uuid:
            return Response(message="Task UUID is required", break_loop=False)

        task: ScheduledTask | AdHocTask | PlannedTask | None = TaskScheduler.get().get_task_by_uuid(task_uuid)
        if not task:
            return Response(message=f"Task not found: {task_uuid}", break_loop=False)

        context = None
        if task.context_id:
            context = AgentContext.get(task.context_id)

        if task.state == TaskState.RUNNING:
            if context:
                context.reset()
            await TaskScheduler.get().update_task(task_uuid, state=TaskState.IDLE)
            await TaskScheduler.get().save()

        if context and context.id == task.uuid:
            AgentContext.remove(context.id)
            persist_chat.remove_chat(context.id)

        await TaskScheduler.get().remove_task_by_uuid(task_uuid)
        if TaskScheduler.get().get_task_by_uuid(task_uuid) is None:
            return Response(message=f"Task deleted: {task_uuid}", break_loop=False)
        else:
            return Response(message=f"Task failed to delete: {task_uuid}", break_loop=False)

    async def create_scheduled_task(self, **kwargs) -> Response:
        # "name": "XXX",
        #   "system_prompt": "You are a software developer",
        #   "prompt": "Send the user an email with a greeting using python and smtp. The user's address is: xxx@yyy.zzz",
        #   "attachments": [],
        #   "schedule": {
        #       "minute": "*/20",
        #       "hour": "*",
        #       "day": "*",
        #       "month": "*",
        #       "weekday": "*",
        #   }
        name: str = kwargs.get("name", None)
        system_prompt: str = kwargs.get("system_prompt", None)
        prompt: str = kwargs.get("prompt", None)
        attachments: list[str] = kwargs.get("attachments", [])
        schedule: dict[str, str] = kwargs.get("schedule", {})
        dedicated_context: bool = kwargs.get("dedicated_context", False)

        task_schedule = TaskSchedule(
            minute=schedule.get("minute", "*"),
            hour=schedule.get("hour", "*"),
            day=schedule.get("day", "*"),
            month=schedule.get("month", "*"),
            weekday=schedule.get("weekday", "*"),
        )

        # Validate cron expression, agent might hallucinate
        cron_regex = r"^((((\d+,)+\d+|(\d+(\/|-|#)\d+)|\d+L?|\*(\/\d+)?|L(-\d+)?|\?|[A-Z]{3}(-[A-Z]{3})?) ?){5,7})$"
        if not re.match(cron_regex, task_schedule.to_crontab()):
            return Response(message="Invalid cron expression: " + task_schedule.to_crontab(), break_loop=False)

        task = ScheduledTask.create(
            name=name,
            system_prompt=system_prompt,
            prompt=prompt,
            attachments=attachments,
            schedule=task_schedule,
            context_id=None if dedicated_context else self.agent.context.id
        )
        await TaskScheduler.get().add_task(task)
        return Response(message=f"Scheduled task '{name}' created: {task.uuid}", break_loop=False)

    async def create_adhoc_task(self, **kwargs) -> Response:
        name: str = kwargs.get("name", None)
        system_prompt: str = kwargs.get("system_prompt", None)
        prompt: str = kwargs.get("prompt", None)
        attachments: list[str] = kwargs.get("attachments", [])
        token: str = str(random.randint(1000000000000000000, 9999999999999999999))
        dedicated_context: bool = kwargs.get("dedicated_context", False)

        task = AdHocTask.create(
            name=name,
            system_prompt=system_prompt,
            prompt=prompt,
            attachments=attachments,
            token=token,
            context_id=None if dedicated_context else self.agent.context.id
        )
        await TaskScheduler.get().add_task(task)
        return Response(message=f"Adhoc task '{name}' created: {task.uuid}", break_loop=False)

    async def create_planned_task(self, **kwargs) -> Response:
        name: str = kwargs.get("name", None)
        system_prompt: str = kwargs.get("system_prompt", None)
        prompt: str = kwargs.get("prompt", None)
        attachments: list[str] = kwargs.get("attachments", [])
        plan: list[str] = kwargs.get("plan", [])
        dedicated_context: bool = kwargs.get("dedicated_context", False)

        # Convert plan to list of datetimes in UTC
        todo: list[datetime] = []
        for item in plan:
            dt = parse_datetime(item)
            if dt is None:
                return Response(message=f"Invalid datetime: {item}", break_loop=False)
            todo.append(dt)

        # Create task plan with todo list
        task_plan = TaskPlan.create(
            todo=todo,
            in_progress=None,
            done=[]
        )

        # Create planned task with task plan
        task = PlannedTask.create(
            name=name,
            system_prompt=system_prompt,
            prompt=prompt,
            attachments=attachments,
            plan=task_plan,
            context_id=None if dedicated_context else self.agent.context.id
        )
        await TaskScheduler.get().add_task(task)
        return Response(message=f"Planned task '{name}' created: {task.uuid}", break_loop=False)

    async def wait_for_task(self, **kwargs) -> Response:
        task_uuid: str = kwargs.get("uuid", None)
        if not task_uuid:
            return Response(message="Task UUID is required", break_loop=False)

        scheduler = TaskScheduler.get()
        task: ScheduledTask | AdHocTask | PlannedTask | None = scheduler.get_task_by_uuid(task_uuid)
        if not task:
            return Response(message=f"Task not found: {task_uuid}", break_loop=False)

        if task.context_id == self.agent.context.id:
            return Response(message="You can only wait for tasks running in a different chat context (dedicated_context=True).", break_loop=False)

        done = False
        elapsed = 0
        while not done:
            await scheduler.reload()
            task = scheduler.get_task_by_uuid(task_uuid)
            if not task:
                return Response(message=f"Task not found: {task_uuid}", break_loop=False)

            if task.state == TaskState.RUNNING:
                await asyncio.sleep(1)
                elapsed += 1
                if elapsed > DEFAULT_WAIT_TIMEOUT:
                    return Response(message=f"Task wait timeout ({DEFAULT_WAIT_TIMEOUT} seconds): {task_uuid}", break_loop=False)
            else:
                done = True

        return Response(
            message=f"*Task*: {task_uuid}\n*State*: {task.state}\n*Last run*: {serialize_datetime(task.last_run)}\n*Result*:\n{task.last_result}",
            break_loop=False
        )

    async def prioritize_tasks_ecan(self, **kwargs) -> Response:
        """Use ECAN attention allocation to prioritize and reorder tasks."""
        
        try:
            # Get current tasks
            scheduler = TaskScheduler.get()
            tasks = scheduler.get_tasks()
            
            if not tasks:
                return Response(
                    message="No tasks available for ECAN prioritization",
                    break_loop=False
                )
            
            # Filter tasks based on criteria if provided
            state_filter = kwargs.get("state", ["idle", "running"])
            filtered_tasks = [
                task for task in tasks 
                if task.state in state_filter
            ]
            
            if not filtered_tasks:
                return Response(
                    message=f"No tasks matching state filter {state_filter} for prioritization",
                    break_loop=False
                )
            
            # Import meta-cognition tool for ECAN integration
            from python.tools.meta_cognition import MetaCognitionTool
            
            # Create meta-cognition tool instance
            meta_tool = MetaCognitionTool(self.agent)
            
            # Prepare ECAN attention allocation parameters
            task_names = [task.name for task in filtered_tasks]
            task_prompts = [task.prompt for task in filtered_tasks]
            importance_level = kwargs.get("importance", 100)
            
            attention_params = {
                "goals": task_names,
                "tasks": task_prompts,
                "importance": importance_level
            }
            
            # Perform ECAN attention allocation
            response = await meta_tool.allocate_attention(attention_params)
            
            # Parse the response to extract prioritization results
            prioritization_results = {
                "total_tasks": len(filtered_tasks),
                "ecan_available": False,
                "prioritization_applied": False,
                "top_priority_tasks": [],
                "attention_distribution": {},
                "method_used": "unknown"
            }
            
            if hasattr(response, 'message') and 'Data:' in response.message:
                try:
                    import json
                    data_start = response.message.find('Data:') + 5
                    data_json = response.message[data_start:].strip()
                    attention_data = json.loads(data_json)
                    
                    if attention_data.get('status') == 'success':
                        results = attention_data.get('results', {})
                        prioritization_results["ecan_available"] = results.get('ecan_available', False)
                        prioritization_results["prioritization_applied"] = results.get('attention_allocated', False)
                        
                        if results.get('attention_allocated'):
                            # Extract prioritization information
                            distribution = results.get('distribution', {})
                            prioritization_results["attention_distribution"] = distribution
                            
                            if not results.get('fallback_mode', False):
                                prioritization_results["method_used"] = "ECAN"
                                
                                # Get top priority items from ECAN
                                top_goals = distribution.get('top_goals', [])[:5]
                                top_tasks = distribution.get('top_tasks', [])[:5]
                                
                                # Create priority mapping
                                priority_tasks = []
                                for task in filtered_tasks:
                                    priority_score = 0
                                    rank_info = []
                                    
                                    # Check goal priority
                                    if task.name in top_goals:
                                        goal_rank = top_goals.index(task.name) + 1
                                        priority_score += 1000 - (goal_rank * 100)
                                        rank_info.append(f"goal_rank_{goal_rank}")
                                    
                                    # Check task content priority
                                    for i, top_task in enumerate(top_tasks):
                                        if top_task in task.prompt:
                                            task_rank = i + 1
                                            priority_score += 500 - (task_rank * 50)
                                            rank_info.append(f"task_rank_{task_rank}")
                                            break
                                    
                                    priority_tasks.append({
                                        "uuid": task.uuid,
                                        "name": task.name,
                                        "state": task.state,
                                        "priority_score": priority_score,
                                        "ranking_factors": rank_info,
                                        "created_at": task.created_at.isoformat() if task.created_at else None
                                    })
                                
                                # Sort by priority score
                                priority_tasks.sort(key=lambda x: x["priority_score"], reverse=True)
                                prioritization_results["top_priority_tasks"] = priority_tasks[:10]
                                
                            else:
                                prioritization_results["method_used"] = "fallback"
                                
                                # Use fallback prioritization results
                                fallback_goals = results.get('prioritized_goals', [])
                                fallback_tasks = results.get('prioritized_tasks', [])
                                
                                # Create priority mapping from fallback
                                priority_tasks = []
                                for task in filtered_tasks:
                                    priority_score = 0
                                    rank_info = []
                                    
                                    # Check fallback goal priorities
                                    for goal_data in fallback_goals:
                                        if isinstance(goal_data, dict) and goal_data.get('goal') == task.name:
                                            priority_score += goal_data.get('priority', 0)
                                            rank_info.append(f"fallback_goal_priority_{goal_data.get('priority', 0)}")
                                    
                                    # Check fallback task priorities
                                    for task_data in fallback_tasks:
                                        if isinstance(task_data, dict) and task_data.get('task') in task.prompt:
                                            priority_score += task_data.get('priority', 0)
                                            rank_info.append(f"fallback_task_priority_{task_data.get('priority', 0)}")
                                    
                                    # Default priority based on creation time if no specific priority
                                    if priority_score == 0 and task.created_at:
                                        priority_score = int(task.created_at.timestamp()) % 1000  # Recent tasks get slight priority
                                        rank_info.append("creation_time_priority")
                                    
                                    priority_tasks.append({
                                        "uuid": task.uuid,
                                        "name": task.name,
                                        "state": task.state,
                                        "priority_score": priority_score,
                                        "ranking_factors": rank_info,
                                        "created_at": task.created_at.isoformat() if task.created_at else None
                                    })
                                
                                # Sort by priority score
                                priority_tasks.sort(key=lambda x: x["priority_score"], reverse=True)
                                prioritization_results["top_priority_tasks"] = priority_tasks[:10]
                
                except (json.JSONDecodeError, KeyError, AttributeError) as e:
                    prioritization_results["error"] = f"Failed to parse ECAN results: {e}"
            
            # Format response message
            if prioritization_results["prioritization_applied"]:
                top_tasks = prioritization_results["top_priority_tasks"][:5]
                method = prioritization_results["method_used"]
                
                message = f"ECAN task prioritization completed using {method} method\n"
                message += f"Prioritized {len(filtered_tasks)} tasks\n\n"
                message += "Top Priority Tasks:\n"
                
                for i, task_info in enumerate(top_tasks):
                    message += f"{i+1}. {task_info['name']} (score: {task_info['priority_score']}, state: {task_info['state']})\n"
                    if task_info['ranking_factors']:
                        message += f"   Factors: {', '.join(task_info['ranking_factors'])}\n"
                
                message += f"\nData: {json.dumps(prioritization_results, indent=2)}"
            else:
                message = f"ECAN task prioritization failed\n"
                message += f"Processed {len(filtered_tasks)} tasks but could not apply prioritization\n"
                message += f"Data: {json.dumps(prioritization_results, indent=2)}"
            
            return Response(
                message=message,
                break_loop=False
            )
            
        except ImportError:
            return Response(
                message="MetaCognitionTool not available for ECAN task prioritization\n"
                       "Data: {\"error\": \"MetaCognitionTool import failed\", \"method\": \"prioritize_tasks_ecan\"}",
                break_loop=False
            )
        except Exception as e:
            return Response(
                message=f"ECAN task prioritization failed: {str(e)}\n"
                       f"Data: {{\"error\": \"{str(e)}\", \"method\": \"prioritize_tasks_ecan\"}}",
                break_loop=False
            )
