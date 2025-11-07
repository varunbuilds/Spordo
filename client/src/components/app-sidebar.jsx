import * as React from "react";
import {
  ArrowUpCircleIcon,
  BarChartIcon,
  Brain,
  CameraIcon,
  ClipboardListIcon,
  DatabaseIcon,
  FileCodeIcon,
  FileIcon,
  FileTextIcon,
  FolderIcon,
  HelpCircleIcon,
  LayoutDashboardIcon,
  ListIcon,
  SearchIcon,
  SettingsIcon,
  TrendingUpIcon,
  UsersIcon,
} from "lucide-react";

import { NavDocuments } from "@/components/nav-documents";
import { NavMain } from "@/components/nav-main";
import { NavSecondary } from "@/components/nav-secondary";
import { NavUser } from "@/components/nav-user";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";

const data = {
  user: {
    name: "Varun Rewadi",
    email: "varunrewadi@spordo.com",
    avatar: "/logo/logo.png",
  },
  navMain: [
    {
      title: "Dashboard",
      url: "/dashboard",
      icon: LayoutDashboardIcon,
    },
    {
      title: "Sports Coach",
      url: "/sports-coach",
      icon: Brain,
    },
    {
      title: "Live Predictions",
      url: "/live-predictions",
      icon: TrendingUpIcon,
    },
    {
      title: "Batting Analysis",
      url: "#",
      icon: BarChartIcon,
    },
    {
      title: "Team Management",
      url: "#",
      icon: UsersIcon,
    },
    {
      title: "Reports",
      url: "#",
      icon: ClipboardListIcon,
    },
    {
      title: "Settings",
      url: "#",
      icon: SettingsIcon,
    },
  ],
  navSecondary: [
    {
      title: "Help Center",
      url: "#",
      icon: HelpCircleIcon,
    },
    {
      title: "Search",
      url: "#",
      icon: SearchIcon,
    },
  ],
  documents: [
    {
      name: "Player Database",
      url: "#",
      icon: DatabaseIcon,
    },
    {
      name: "Match Reports",
      url: "#",
      icon: FileTextIcon,
    },
    {
      name: "Training Plans",
      url: "#",
      icon: FileIcon,
    },
  ],
};

export function AppSidebar({ ...props }) {
  return (
    <Sidebar collapsible="offcanvas" {...props}>
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              asChild
              className="data-[slot=sidebar-menu-button]:!p-1.5"
            >
              <a href="/">
                <img
                  src="/logo/logo.svg"
                  alt="Spordo"
                  className="h-8 w-8 filter invert"
                />
                <span className="text-base font-semibold">SPORDO</span>
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <NavMain items={data.navMain} />
        <NavDocuments items={data.documents} />
        <NavSecondary items={data.navSecondary} className="mt-auto" />
      </SidebarContent>
      <SidebarFooter>
        <NavUser user={data.user} />
      </SidebarFooter>
    </Sidebar>
  );
}
